from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import imageio.v2 as imageio
import imageio_ffmpeg
import matplotlib
import numpy as np
import soundfile as sf
from matplotlib.colors import Colormap, LinearSegmentedColormap
from scipy import signal
from scipy.fft import rfft, rfftfreq

matplotlib.use("TkAgg")

import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


ProgressCallback = Callable[[str, int, int], None]


class WorkCancelled(RuntimeError):
	"""Raised when a background operation is cancelled by the user."""


@dataclass(frozen=True)
class AnalysisSettings:
	fps: int = 24
	fft_size: int = 12_000
	low_frequency: float = 30.0
	high_frequency: float = 12_000.0
	chunk_frames: int = 192


@dataclass(frozen=True)
class EQSettings:
	low_gain_db: float = 0.0
	bell_gain_db: float = 0.0
	high_gain_db: float = 0.0
	low_frequency: float = 120.0
	bell_frequency: float = 1_000.0
	high_frequency: float = 5_000.0
	bell_width_octaves: float = 1.25

	def gain_db(self, frequencies: np.ndarray) -> np.ndarray:
		safe_frequencies = np.maximum(frequencies.astype(np.float64), 1.0)
		low_weight = 1.0 / (1.0 + (safe_frequencies / self.low_frequency) ** 4)
		high_weight = 1.0 / (1.0 + (self.high_frequency / safe_frequencies) ** 4)
		octave_distance = np.log2(safe_frequencies / self.bell_frequency)
		bell_weight = np.exp(-0.5 * (octave_distance / self.bell_width_octaves) ** 2)
		return (
			self.low_gain_db * low_weight
			+ self.bell_gain_db * bell_weight
			+ self.high_gain_db * high_weight
		)

	def linear_gain(self, frequencies: np.ndarray) -> np.ndarray:
		return np.power(10.0, self.gain_db(frequencies) / 20.0).astype(np.float32)


@dataclass(frozen=True)
class ProcessingSettings:
	eq: EQSettings = EQSettings()
	outlier_iqr_multiplier: float = 1.5
	clip_fraction: float = 0.25
	contrast: float = 0.0
	compression_threshold: float = 0.7
	compression_amount: float = 0.5
	smoothing: float = 0.3


@dataclass
class AudioAnalysis:
	source_path: Path
	audio: np.ndarray
	sample_rate: int
	frequencies: np.ndarray
	spectra: np.ndarray
	fps: int
	fft_size: int

	@property
	def duration(self) -> float:
		return self.audio.shape[0] / self.sample_rate

	@property
	def frame_count(self) -> int:
		return self.spectra.shape[0]


@dataclass(frozen=True)
class ChannelNormalization:
	frame_scales: np.ndarray
	frame_maxima: np.ndarray
	high_threshold: float
	target_maximum: float
	coefficient_min: float
	coefficient_max: float
	outlier_count: int


@dataclass(frozen=True)
class ProcessingState:
	settings: ProcessingSettings
	eq_gain: np.ndarray
	channel_normalizations: tuple[ChannelNormalization, ChannelNormalization]


@dataclass
class ProcessedAudio:
	analysis: AudioAnalysis
	spectra: np.ndarray
	state: ProcessingState


@dataclass
class ExportResult:
	path: Path
	preview_frames: np.ndarray | None = None
	preview_fps: float = 0.0


def _report(progress: ProgressCallback | None, phase: str, current: int, total: int) -> None:
	if progress is not None:
		progress(phase, current, total)


def _check_cancelled(cancel_event: threading.Event | None) -> None:
	if cancel_event is not None and cancel_event.is_set():
		raise WorkCancelled("Operation cancelled")


def analyze_audio(
	source_path: str | Path,
	settings: AnalysisSettings,
	progress: ProgressCallback | None = None,
	cancel_event: threading.Event | None = None,
) -> AudioAnalysis:
	"""Load audio and compute stereo FFT magnitudes in bounded-memory chunks."""
	path = Path(source_path).expanduser().resolve()
	audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
	if audio.shape[1] == 1:
		audio = np.repeat(audio, 2, axis=1)
	elif audio.shape[1] > 2:
		audio = audio[:, :2]

	fft_size = max(256, int(settings.fft_size))
	fps = max(1, int(settings.fps))
	high_frequency = min(float(settings.high_frequency), sample_rate / 2.0)
	if settings.low_frequency >= high_frequency:
		raise ValueError("The low frequency must be below the high frequency and Nyquist limit.")

	all_frequencies = rfftfreq(fft_size, d=1.0 / sample_rate)
	frequency_mask = (
		(all_frequencies >= float(settings.low_frequency))
		& (all_frequencies <= high_frequency)
	)
	frequencies = all_frequencies[frequency_mask].astype(np.float32)
	if frequencies.size == 0:
		raise ValueError("The selected frequency range contains no FFT bins.")

	frame_count = max(1, math.ceil(audio.shape[0] / sample_rate * fps))
	frame_positions = (np.arange(frame_count, dtype=np.float64) * sample_rate / fps).astype(np.int64)
	padded_audio = np.pad(audio, ((0, fft_size), (0, 0)))
	sample_offsets = np.arange(fft_size, dtype=np.int64)
	window = signal.windows.hann(fft_size, sym=False).astype(np.float32)
	spectra = np.empty((frame_count, frequencies.size, 2), dtype=np.float32)

	chunk_size = max(1, int(settings.chunk_frames))
	for chunk_start in range(0, frame_count, chunk_size):
		_check_cancelled(cancel_event)
		chunk_end = min(frame_count, chunk_start + chunk_size)
		indices = frame_positions[chunk_start:chunk_end, None] + sample_offsets[None, :]
		segments = padded_audio[indices]
		segments *= window[None, :, None]
		transformed = rfft(segments, axis=1, workers=-1)
		spectra[chunk_start:chunk_end] = np.abs(transformed[:, frequency_mask, :]).astype(
			np.float32
		)
		_report(progress, "Analyzing audio", chunk_end, frame_count)

	return AudioAnalysis(
		source_path=path,
		audio=np.ascontiguousarray(audio),
		sample_rate=int(sample_rate),
		frequencies=frequencies,
		spectra=spectra,
		fps=fps,
		fft_size=fft_size,
	)


def _outlier_normalize(
	channel: np.ndarray, iqr_multiplier: float
) -> tuple[np.ndarray, ChannelNormalization]:
	frame_maxima = np.max(channel, axis=1)
	q1, q3 = np.percentile(frame_maxima, [25.0, 96.0])
	iqr_multiplier = max(0.0, float(iqr_multiplier))
	high_threshold = float(q3 + iqr_multiplier * (q3 - q1))
	outlier_mask = frame_maxima > high_threshold
	non_outlier_maxima = frame_maxima[~outlier_mask]
	target_maximum = float(np.mean(non_outlier_maxima)) if non_outlier_maxima.size else 0.0

	frame_scales = np.ones(frame_maxima.shape, dtype=np.float32)
	scalable = outlier_mask & (frame_maxima > 0.0)
	frame_scales[scalable] = target_maximum / frame_maxima[scalable]
	normalized = channel * frame_scales[:, None]

	coefficient_min = float(np.min(normalized))
	coefficient_max = float(np.max(normalized))
	dynamic_range = coefficient_max - coefficient_min
	if dynamic_range > 0.0:
		normalized = (normalized - coefficient_min) / dynamic_range
	else:
		normalized = np.zeros_like(normalized)

	state = ChannelNormalization(
		frame_scales=frame_scales,
		frame_maxima=frame_maxima,
		high_threshold=high_threshold,
		target_maximum=target_maximum,
		coefficient_min=coefficient_min,
		coefficient_max=coefficient_max,
		outlier_count=int(np.count_nonzero(outlier_mask)),
	)
	return normalized.astype(np.float32, copy=False), state


def clip_small_coefficients(spectra: np.ndarray, fraction: float) -> np.ndarray:
	fraction = float(np.clip(fraction, 0.0, 0.99))
	frame_maxima = np.max(spectra, axis=1, keepdims=True)
	return np.where(spectra < fraction * frame_maxima, 0.0, spectra).astype(
		np.float32, copy=False
	)


def apply_contrast(spectra: np.ndarray, contrast: float) -> np.ndarray:
	"""Apply the requested continuous family between a flat field and a binary step."""
	contrast = float(np.clip(contrast, -1.0, 1.0))
	if abs(contrast) < 1e-9:
		return spectra.copy()

	frame_maxima = np.max(spectra, axis=1, keepdims=True)
	frame_minima = np.min(spectra, axis=1, keepdims=True)

	if contrast < 0.0:
		amount = -contrast
		adjusted = (1.0 - amount) * spectra + amount * frame_maxima
	else:
		midpoint = 0.5 * (frame_minima + frame_maxima)
		step_target = np.where(spectra < midpoint, frame_minima, frame_maxima)
		adjusted = (1.0 - contrast) * spectra + contrast * step_target

	return adjusted.astype(np.float32, copy=False)


def apply_dynamic_compression(
	spectra: np.ndarray, threshold_fraction: float, amount: float
) -> np.ndarray:
	threshold_fraction = float(np.clip(threshold_fraction, 0.0, 1.0))
	amount = float(np.clip(amount, 0.0, 1.0))
	compressed = spectra.copy()

	for channel_index in range(compressed.shape[2]):
		channel = compressed[:, :, channel_index]
		channel_maximum = float(np.max(channel))
		threshold = threshold_fraction * channel_maximum
		above_threshold = channel > threshold
		channel[above_threshold] = threshold + (1.0 - amount) * (
			channel[above_threshold] - threshold
		)
		result_minimum = float(np.min(channel))
		result_maximum = float(np.max(channel))
		if result_maximum > result_minimum:
			channel -= result_minimum
			channel /= result_maximum - result_minimum

	return compressed


def apply_temporal_smoothing(spectra: np.ndarray, smoothing: float) -> np.ndarray:
	smoothing = float(np.clip(smoothing, 0.0, 1.0))
	if smoothing <= 0.0 or spectra.shape[0] < 2:
		return spectra.copy()
	initial_state = (smoothing * spectra[0])[None, :, :]
	smoothed, _ = signal.lfilter(
		[1.0 - smoothing],
		[1.0, -smoothing],
		spectra,
		axis=0,
		zi=initial_state,
	)
	return smoothed.astype(np.float32, copy=False)


def process_audio(
	analysis: AudioAnalysis,
	settings: ProcessingSettings,
	progress: ProgressCallback | None = None,
	cancel_event: threading.Event | None = None,
) -> ProcessedAudio:
	"""Run cached FFT magnitudes through EQ, clipping, contrast, compression, and EMA."""
	_check_cancelled(cancel_event)
	eq_gain = settings.eq.linear_gain(analysis.frequencies)
	working = analysis.spectra * eq_gain[None, :, None]
	_report(progress, "Applying EQ", 1, 5)

	normalizations: list[ChannelNormalization] = []
	for channel_index in range(2):
		working[:, :, channel_index], normalization = _outlier_normalize(
			working[:, :, channel_index], settings.outlier_iqr_multiplier
		)
		normalizations.append(normalization)
	_report(progress, "Clipping outliers", 2, 5)
	_check_cancelled(cancel_event)

	working = clip_small_coefficients(working, settings.clip_fraction)
	working = apply_contrast(working, settings.contrast)
	_report(progress, "Shaping contrast", 3, 5)
	_check_cancelled(cancel_event)

	working = apply_dynamic_compression(
		working,
		settings.compression_threshold,
		settings.compression_amount,
	)
	_report(progress, "Compressing dynamics", 4, 5)
	_check_cancelled(cancel_event)

	working = apply_temporal_smoothing(working, settings.smoothing)
	_report(progress, "Smoothing", 5, 5)
	state = ProcessingState(
		settings=settings,
		eq_gain=eq_gain,
		channel_normalizations=(normalizations[0], normalizations[1]),
	)
	return ProcessedAudio(analysis=analysis, spectra=working, state=state)


def diagnostic_clipping_frame(
	processed: ProcessedAudio, frame_index: int
) -> tuple[np.ndarray, np.ndarray]:
	"""Reconstruct one frame immediately before and after small-coefficient clipping."""
	analysis = processed.analysis
	frame_index = int(np.clip(frame_index, 0, analysis.frame_count - 1))
	before = np.empty((analysis.frequencies.size, 2), dtype=np.float32)

	for channel_index, normalization in enumerate(processed.state.channel_normalizations):
		values = (
			analysis.spectra[frame_index, :, channel_index]
			* processed.state.eq_gain
			* normalization.frame_scales[frame_index]
		)
		dynamic_range = normalization.coefficient_max - normalization.coefficient_min
		if dynamic_range > 0.0:
			values = (values - normalization.coefficient_min) / dynamic_range
		else:
			values = np.zeros_like(values)
		before[:, channel_index] = np.clip(values, 0.0, 1.0)

	maxima = np.max(before, axis=0, keepdims=True)
	after = np.where(
		before < processed.state.settings.clip_fraction * maxima,
		0.0,
		before,
	).astype(np.float32)
	return before, after


class CartesianWaveRenderer:
	"""Synthesize the notebook's separable cosine field through a fast real FFT series."""

	def __init__(self, width: int, height: int, frequencies: np.ndarray) -> None:
		self.width = max(64, int(width))
		self.height = max(64, int(height))
		minimum_period = 16 * (self.width + self.height)
		self.period = 1 << (minimum_period - 1).bit_length()

		spatial_low = 1.0 / self.height
		spatial_high = 1.0 / 8.0
		spatial_frequencies = np.linspace(
			spatial_low,
			spatial_high,
			frequencies.size,
			dtype=np.float64,
		)
		self.harmonic_indices = np.clip(
			np.rint(spatial_frequencies * self.period).astype(np.int32),
			1,
			self.period // 2 - 1,
		)

		x_values = np.arange(self.width, dtype=np.int32)[None, :]
		y_values = np.arange(self.height, dtype=np.int32)[:, None]
		self.sum_indices = np.asarray((x_values + y_values) % self.period, dtype=np.int32)
		self.difference_indices = np.asarray((x_values - y_values) % self.period, dtype=np.int32)

	def _cosine_series(self, coefficients: np.ndarray) -> np.ndarray:
		weights = np.bincount(
			self.harmonic_indices,
			weights=coefficients,
			minlength=self.period // 2 + 1,
		)
		return (np.fft.irfft(weights, n=self.period) * (self.period / 2.0)).astype(
			np.float32
		)

	def intensity(self, left_coefficients: np.ndarray, right_coefficients: np.ndarray) -> np.ndarray:
		left_series = self._cosine_series(left_coefficients)
		right_series = self._cosine_series(right_coefficients)
		left_field = 0.5 * (
			left_series[self.sum_indices] + left_series[self.difference_indices]
		)
		right_field = 0.5 * (
			right_series[self.sum_indices] + right_series[self.difference_indices]
		)
		left_field += right_field[:, ::-1]
		np.abs(left_field, out=left_field)
		return left_field


def build_colormap(name: str, custom_colors: Sequence[str]) -> Colormap:
	if name == "Custom":
		if len(custom_colors) < 2:
			raise ValueError("A custom colormap needs at least two colors.")
		return LinearSegmentedColormap.from_list("custom_visualizer", list(custom_colors), N=256)
	return matplotlib.colormaps[name]


def colormap_lut(colormap: Colormap) -> np.ndarray:
	samples = colormap(np.linspace(0.0, 1.0, 256))[:, :3]
	return np.rint(samples * 255.0).astype(np.uint8)


def colorize_intensity(
	intensity: np.ndarray,
	normalization_reference: float,
	colormap_upper: float,
	lut: np.ndarray,
) -> np.ndarray:
	if normalization_reference <= 0.0:
		normalized = np.zeros_like(intensity)
	else:
		normalized = intensity / normalization_reference

	upper = float(np.clip(colormap_upper, 0.0, 1.0))
	if upper <= 0.0:
		indices = np.full(intensity.shape, 255, dtype=np.uint8)
	else:
		indices = np.asarray(np.clip(normalized / upper, 0.0, 1.0) * 255.0, dtype=np.uint8)
	return lut[indices]


def render_frame_rgb(
	processed: ProcessedAudio,
	frame_index: int,
	renderer: CartesianWaveRenderer,
	colormap: Colormap,
	colormap_upper: float,
	normalization_reference: float | None = None,
) -> np.ndarray:
	frame_index = int(np.clip(frame_index, 0, processed.analysis.frame_count - 1))
	coefficients = processed.spectra[frame_index]
	intensity = renderer.intensity(coefficients[:, 0], coefficients[:, 1])
	if normalization_reference is None:
		normalization_reference = float(np.max(np.sum(processed.spectra, axis=(1, 2))))
	return colorize_intensity(
		intensity,
		normalization_reference,
		colormap_upper,
		colormap_lut(colormap),
	)


def export_video(
	processed: ProcessedAudio,
	output_path: str | Path,
	width: int,
	height: int,
	colormap: Colormap,
	colormap_upper: float,
	start_seconds: float = 0.0,
	duration_seconds: float | None = None,
	progress: ProgressCallback | None = None,
	cancel_event: threading.Event | None = None,
) -> Path:
	"""Stream rendered frames to H.264, then mux the matching source audio."""
	analysis = processed.analysis
	output = Path(output_path).expanduser().resolve()
	output.parent.mkdir(parents=True, exist_ok=True)
	start_frame = int(np.clip(round(start_seconds * analysis.fps), 0, analysis.frame_count - 1))
	if duration_seconds is None:
		end_frame = analysis.frame_count
	else:
		end_frame = min(
			analysis.frame_count,
			start_frame + max(1, round(duration_seconds * analysis.fps)),
		)
	frame_indices = range(start_frame, end_frame)
	total_frames = len(frame_indices)
	renderer = CartesianWaveRenderer(width, height, analysis.frequencies)
	lut = colormap_lut(colormap)
	normalization_reference = float(np.max(np.sum(processed.spectra, axis=(1, 2))))

	descriptor, temporary_name = tempfile.mkstemp(
		prefix="auviz_", suffix="_video_only.mp4", dir=output.parent
	)
	os.close(descriptor)
	temporary_video = Path(temporary_name)
	writer = None
	try:
		writer = imageio.get_writer(
			temporary_video,
			fps=analysis.fps,
			codec="libx264",
			pixelformat="yuv420p",
			quality=8,
			macro_block_size=1,
			ffmpeg_log_level="error",
		)
		for completed, frame_index in enumerate(frame_indices, start=1):
			_check_cancelled(cancel_event)
			coefficients = processed.spectra[frame_index]
			intensity = renderer.intensity(coefficients[:, 0], coefficients[:, 1])
			rgb = colorize_intensity(
				intensity,
				normalization_reference,
				colormap_upper,
				lut,
			)
			writer.append_data(rgb)
			_report(progress, "Rendering video", completed, total_frames)
		writer.close()
		writer = None

		_check_cancelled(cancel_event)
		ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
		audio_input: list[str] = []
		if start_seconds > 0.0:
			audio_input.extend(["-ss", f"{start_seconds:.6f}"])
		if duration_seconds is not None:
			audio_input.extend(["-t", f"{duration_seconds:.6f}"])
		audio_input.extend(["-i", str(analysis.source_path)])
		command = [
			ffmpeg_path,
			"-y",
			"-i",
			str(temporary_video),
			*audio_input,
			"-map",
			"0:v:0",
			"-map",
			"1:a:0",
			"-c:v",
			"copy",
			"-c:a",
			"aac",
			"-shortest",
			str(output),
		]
		result = subprocess.run(command, capture_output=True, text=True, check=False)
		if result.returncode != 0:
			raise RuntimeError(f"ffmpeg could not mux the audio:\n{result.stderr[-2000:]}")
		_report(progress, "Video complete", total_frames, total_frames)
		return output
	finally:
		if writer is not None:
			writer.close()
		temporary_video.unlink(missing_ok=True)


def decode_video_preview(
	video_path: str | Path,
	width: int,
	height: int,
	cancel_event: threading.Event | None = None,
) -> np.ndarray:
	"""Decode an exported video into compact RGB frames for the embedded player."""
	frame_bytes = width * height * 3
	command = [
		imageio_ffmpeg.get_ffmpeg_exe(),
		"-loglevel",
		"error",
		"-i",
		str(video_path),
		"-an",
		"-vf",
		f"scale={width}:{height}",
		"-pix_fmt",
		"rgb24",
		"-f",
		"rawvideo",
		"-",
	]
	process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	frames: list[np.ndarray] = []
	try:
		if process.stdout is None:
			raise RuntimeError("ffmpeg did not provide a video stream.")
		while True:
			_check_cancelled(cancel_event)
			data = process.stdout.read(frame_bytes)
			if not data:
				break
			if len(data) != frame_bytes:
				raise RuntimeError("The sample preview ended with an incomplete frame.")
			frame = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3).copy()
			frames.append(frame)
		return_code = process.wait()
		if return_code != 0:
			stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
			raise RuntimeError(f"Could not decode the sample preview:\n{stderr[-2000:]}")
	finally:
		if process.poll() is None:
			process.terminate()
			process.wait()
	if not frames:
		raise RuntimeError("The generated sample contains no video frames.")
	return np.stack(frames)


class ColormapPicker(ttk.Frame):
	"""A searchable dropdown whose rows include colormap gradient previews."""

	def __init__(
		self,
		parent: tk.Misc,
		variable: tk.StringVar,
		choices: Sequence[str],
		colormap_for_name: Callable[[str], Colormap],
		on_select: Callable[[], None],
	) -> None:
		super().__init__(parent)
		self.variable = variable
		self.choices = list(choices)
		self.colormap_for_name = colormap_for_name
		self.on_select = on_select
		self.popup: tk.Toplevel | None = None
		self.tree: ttk.Treeview | None = None
		self.search_var = tk.StringVar()
		self.search_var.trace_add("write", lambda *_args: self._populate())
		self.images: dict[str, tk.PhotoImage] = {}
		self.item_names: dict[str, str] = {}
		self.button = ttk.Button(self, textvariable=self.variable, command=self._open)
		self.button.pack(fill=tk.X)

	def set_choices(self, choices: Sequence[str]) -> None:
		self.choices = list(choices)
		self.images.clear()
		if self.popup is not None:
			self._populate()

	def refresh_image(self, name: str) -> None:
		self.images.pop(name, None)
		if self.popup is not None:
			self._populate()

	def _preview_image(self, name: str) -> tk.PhotoImage:
		if name in self.images:
			return self.images[name]
		lut = colormap_lut(self.colormap_for_name(name))
		indices = np.linspace(0, len(lut) - 1, 88).astype(int)
		colors = [f"#{red:02x}{green:02x}{blue:02x}" for red, green, blue in lut[indices]]
		image = tk.PhotoImage(master=self, width=len(colors), height=14)
		row = "{" + " ".join(colors) + "}"
		for y_value in range(14):
			image.put(row, to=(0, y_value))
		self.images[name] = image
		return image

	def _open(self) -> None:
		if self.popup is not None:
			self._close()
			return
		popup = tk.Toplevel(self)
		self.popup = popup
		popup.overrideredirect(True)
		popup.transient(self.winfo_toplevel())
		popup.configure(background="#9aaba9")
		popup.geometry(f"340x420+{self.winfo_rootx()}+{self.winfo_rooty() + self.winfo_height()}")

		container = ttk.Frame(popup, padding=5)
		container.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
		search = ttk.Entry(container, textvariable=self.search_var)
		search.pack(fill=tk.X, pady=(0, 5))
		list_frame = ttk.Frame(container)
		list_frame.pack(fill=tk.BOTH, expand=True)
		tree = ttk.Treeview(list_frame, show="tree", selectmode="browse")
		self.tree = tree
		scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
		tree.configure(yscrollcommand=scrollbar.set)
		tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
		tree.bind("<ButtonRelease-1>", self._choose)
		tree.bind("<Return>", self._choose)
		popup.bind("<Escape>", lambda _event: self._close())
		popup.bind("<FocusOut>", self._focus_out)
		self._populate()
		popup.update_idletasks()
		search.focus_set()

	def _populate(self) -> None:
		if self.tree is None:
			return
		query = self.search_var.get().strip().casefold()
		self.tree.delete(*self.tree.get_children())
		self.item_names.clear()
		selected_item = None
		for index, name in enumerate(self.choices):
			if query and query not in name.casefold():
				continue
			item_id = f"map_{index}"
			self.item_names[item_id] = name
			self.tree.insert("", tk.END, iid=item_id, text=name, image=self._preview_image(name))
			if name == self.variable.get():
				selected_item = item_id
		if selected_item is not None:
			self.tree.selection_set(selected_item)
			self.tree.see(selected_item)

	def _choose(self, _event: object | None = None) -> None:
		if self.tree is None or not self.tree.selection():
			return
		name = self.item_names[self.tree.selection()[0]]
		self.variable.set(name)
		self._close()
		self.on_select()

	def _focus_out(self, _event: object) -> None:
		if self.popup is not None:
			self.after_idle(self._close_if_focus_left)

	def _close_if_focus_left(self) -> None:
		if self.popup is not None and self.popup.focus_get() is None:
			self._close()

	def _close(self) -> None:
		if self.popup is not None:
			self.popup.destroy()
		self.popup = None
		self.tree = None
		self.search_var.set("")


class EQCurveEditor(ttk.Frame):
	"""A compact draggable low-shelf, bell, and high-shelf editor."""

	def __init__(
		self,
		parent: tk.Misc,
		on_change: Callable[[EQSettings], None],
	) -> None:
		super().__init__(parent)
		self.on_change = on_change
		self.low_gain = 0.0
		self.bell_gain = 0.0
		self.high_gain = 0.0
		self.low_frequency = 120.0
		self.bell_frequency = 1_000.0
		self.high_frequency = 5_000.0
		self.bell_width_octaves = 1.25
		self.dragging: str | None = None

		self.figure = Figure(figsize=(3.4, 2.25), dpi=100)
		self.figure.patch.set_facecolor("#f5f7f6")
		self.axis = self.figure.add_subplot(111)
		self.canvas = FigureCanvasTkAgg(self.figure, master=self)
		self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
		self.canvas.mpl_connect("button_press_event", self._on_press)
		self.canvas.mpl_connect("motion_notify_event", self._on_motion)
		self.canvas.mpl_connect("button_release_event", self._on_release)
		self._draw()

	def settings(self) -> EQSettings:
		return EQSettings(
			low_gain_db=self.low_gain,
			bell_gain_db=self.bell_gain,
			high_gain_db=self.high_gain,
			low_frequency=self.low_frequency,
			bell_frequency=self.bell_frequency,
			high_frequency=self.high_frequency,
			bell_width_octaves=self.bell_width_octaves,
		)

	def set_settings(self, settings: EQSettings) -> None:
		self.low_gain = settings.low_gain_db
		self.bell_gain = settings.bell_gain_db
		self.high_gain = settings.high_gain_db
		self.low_frequency = settings.low_frequency
		self.bell_frequency = settings.bell_frequency
		self.high_frequency = settings.high_frequency
		self.bell_width_octaves = settings.bell_width_octaves
		self._draw()

	def reset(self) -> None:
		self.low_gain = 0.0
		self.bell_gain = 0.0
		self.high_gain = 0.0
		self.low_frequency = 120.0
		self.bell_frequency = 1_000.0
		self.high_frequency = 5_000.0
		self.bell_width_octaves = 1.25
		self._draw()
		self.on_change(self.settings())

	def _control_points(self) -> dict[str, tuple[float, float]]:
		return {
			"low": (self.low_frequency, self.low_gain),
			"bell": (self.bell_frequency, self.bell_gain),
			"high": (self.high_frequency, self.high_gain),
		}

	def _draw(self) -> None:
		frequencies = np.geomspace(30.0, 12_000.0, 500)
		curve = self.settings().gain_db(frequencies)
		self.axis.clear()
		self.axis.set_facecolor("#f5f7f6")
		self.axis.axhline(0.0, color="#869496", linewidth=0.8)
		self.axis.plot(frequencies, curve, color="#087f8c", linewidth=2.2)
		self.axis.fill_between(frequencies, 0.0, curve, where=curve >= 0.0, color="#087f8c", alpha=0.12)
		self.axis.fill_between(frequencies, 0.0, curve, where=curve < 0.0, color="#e45756", alpha=0.12)
		for name, (frequency, gain) in self._control_points().items():
			color = "#e45756" if name == "bell" else "#f2a541"
			self.axis.scatter([frequency], [gain], s=52, color=color, edgecolor="#172a3a", zorder=5)
			self.axis.annotate(
				f"{frequency:,.0f} Hz\n{gain:+.1f} dB",
				(frequency, gain),
				xytext=(0, 9),
				textcoords="offset points",
				ha="center",
				fontsize=7,
				color="#172a3a",
			)
		self.axis.set_xscale("log")
		self.axis.set_xlim(30.0, 12_000.0)
		self.axis.set_ylim(-24.0, 24.0)
		self.axis.set_xlabel("Frequency (Hz)", fontsize=8)
		self.axis.set_ylabel("Gain (dB)", fontsize=8)
		self.axis.tick_params(labelsize=7, colors="#40525e")
		self.axis.grid(True, which="both", color="#d9e1df", linewidth=0.6, alpha=0.8)
		for spine in self.axis.spines.values():
			spine.set_color("#b9c6c3")
		self.figure.tight_layout(pad=0.8)
		self.canvas.draw_idle()

	def _nearest_point(self, event: object) -> str | None:
		if getattr(event, "x", None) is None or getattr(event, "y", None) is None:
			return None
		nearest_name = None
		nearest_distance = float("inf")
		for name, point in self._control_points().items():
			pixel_x, pixel_y = self.axis.transData.transform(point)
			distance = math.hypot(event.x - pixel_x, event.y - pixel_y)
			if distance < nearest_distance:
				nearest_name = name
				nearest_distance = distance
		return nearest_name if nearest_distance <= 16.0 else None

	def _on_press(self, event: object) -> None:
		if getattr(event, "inaxes", None) is self.axis:
			self.dragging = self._nearest_point(event)

	def _on_motion(self, event: object) -> None:
		if self.dragging is None or getattr(event, "inaxes", None) is not self.axis:
			return
		y_value = getattr(event, "ydata", None)
		if y_value is None:
			return
		gain = float(np.clip(y_value, -24.0, 24.0))
		x_value = getattr(event, "xdata", None)
		if self.dragging == "low":
			self.low_gain = gain
			if x_value is not None:
				self.low_frequency = float(np.clip(x_value, 40.0, min(1_000.0, self.high_frequency / 1.5)))
		elif self.dragging == "high":
			self.high_gain = gain
			if x_value is not None:
				self.high_frequency = float(np.clip(x_value, max(1_000.0, self.low_frequency * 1.5), 10_000.0))
		else:
			self.bell_gain = gain
			if x_value is not None:
				self.bell_frequency = float(np.clip(x_value, 180.0, 3_500.0))
		self._draw()

	def _on_release(self, _event: object) -> None:
		if self.dragging is None:
			return
		self.dragging = None
		self.on_change(self.settings())


class AudioVisualizerApp:
	PROJECT_FORMAT = "auviz-project"
	PROJECT_VERSION = 1
	PREVIEW_WIDTH = 720
	PREVIEW_HEIGHT = 405
	DEFAULT_SAMPLE_DURATION = 3.0
	MIN_SAMPLE_DURATION = 0.25
	MAX_SAMPLE_DURATION = 15.0

	def __init__(self, root: tk.Tk) -> None:
		self.root = root
		self.root.title("AUVIZ - Wave Bundle Studio")
		self.root.geometry("1480x920")
		self.root.minsize(1180, 760)
		self.root.protocol("WM_DELETE_WINDOW", self._close)

		self.analysis: AudioAnalysis | None = None
		self.processed: ProcessedAudio | None = None
		self.audio_path: Path | None = None
		self.project_path: Path | None = None
		self.last_video: Path | None = None
		self.pending_project_view: dict[str, float] | None = None
		self.processing_dirty = True
		self.normalization_reference = 1.0
		self.preview_renderer: CartesianWaveRenderer | None = None
		self.preview_after_id: str | None = None
		self.sample_span = None
		self.sample_preview_frames: np.ndarray | None = None
		self.sample_preview_path: Path | None = None
		self.sample_preview_fps = 0.0
		self.sample_playback_after_id: str | None = None
		self.sample_playback_started = 0.0
		self.sample_playing = False
		self.sample_audio_process: subprocess.Popen[bytes] | None = None
		self.busy = False
		self.export_active = False

		self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="auviz")
		self.events: queue.Queue[tuple[object, ...]] = queue.Queue()
		self.cancel_event = threading.Event()
		self.action_widgets: list[ttk.Button] = []

		self.custom_colors = ["#081c2c", "#087f8c", "#f2a541", "#e45756", "#f7f3e8"]
		self.builtin_colormap_names = list(
			dict.fromkeys(["turbo", "viridis", "inferno", "magma", "cividis", "RdBu_r", *sorted(matplotlib.colormaps)])
		)
		self.saved_colormaps = self._load_saved_colormaps()
		self._create_variables()
		self._configure_style()
		self._create_layout()
		self._redraw_colormap_swatch()
		self._redraw_palette_buttons()
		self.poll_after_id = self.root.after(80, self._poll_events)

	def _create_variables(self) -> None:
		self.file_name_var = tk.StringVar(value="No audio loaded")
		self.audio_info_var = tk.StringVar(value="")
		self.fps_var = tk.IntVar(value=24)
		self.fft_size_var = tk.IntVar(value=12_000)
		self.low_frequency_var = tk.DoubleVar(value=30.0)
		self.high_frequency_var = tk.DoubleVar(value=12_000.0)

		self.clip_fraction_var = tk.DoubleVar(value=0.25)
		self.outlier_iqr_multiplier_var = tk.DoubleVar(value=1.5)
		self.contrast_var = tk.DoubleVar(value=0.0)
		self.compression_threshold_var = tk.DoubleVar(value=0.7)
		self.compression_amount_var = tk.DoubleVar(value=0.5)
		self.smoothing_var = tk.DoubleVar(value=0.3)

		self.colormap_var = tk.StringVar(value="turbo")
		self.custom_colormap_name_var = tk.StringVar(value="")
		self.colormap_upper_var = tk.DoubleVar(value=0.3)
		self.resolution_var = tk.StringVar(value="1280 x 720")
		self.output_path_var = tk.StringVar(value=str(Path.cwd() / "visualizer.mp4"))

		self.frame_var = tk.DoubleVar(value=0.0)
		self.frame_entry_var = tk.StringVar(value="0")
		self.frame_time_var = tk.StringVar(value="0.00 s")
		self.sample_start_var = tk.DoubleVar(value=0.0)
		self.sample_duration_var = tk.DoubleVar(value=self.DEFAULT_SAMPLE_DURATION)
		self.sample_time_var = tk.StringVar(value="0.00 - 3.00 s")
		self.sample_button_var = tk.StringVar(value="Render 3 s Sample")
		self.sample_preview_var = tk.StringVar(value="No sample rendered")
		self.export_progress_var = tk.StringVar(value="Ready to render")
		self.status_var = tk.StringVar(value="Choose an audio file to begin")

	def _configure_style(self) -> None:
		self.colors = {
			"paper": "#f5f7f6",
			"ink": "#172a3a",
			"muted": "#60727d",
			"teal": "#087f8c",
			"coral": "#e45756",
			"gold": "#f2a541",
			"line": "#cdd8d5",
			"panel": "#edf2f0",
		}
		self.root.configure(background=self.colors["paper"])
		style = ttk.Style(self.root)
		try:
			style.theme_use("clam")
		except tk.TclError:
			pass
		style.configure(".", font=("Avenir Next", 10), background=self.colors["paper"], foreground=self.colors["ink"])
		style.configure("TFrame", background=self.colors["paper"])
		style.configure("Panel.TFrame", background=self.colors["panel"])
		style.configure("TLabel", background=self.colors["paper"], foreground=self.colors["ink"])
		style.configure("Muted.TLabel", foreground=self.colors["muted"])
		style.configure("Title.TLabel", font=("Avenir Next", 20, "bold"), foreground=self.colors["ink"])
		style.configure("Section.TLabel", font=("Avenir Next", 11, "bold"), foreground=self.colors["ink"])
		style.configure("Accent.TButton", background=self.colors["teal"], foreground="white", padding=(10, 7))
		style.map("Accent.TButton", background=[("active", "#066c77"), ("disabled", "#9eb7b7")])
		style.configure("Danger.TButton", foreground=self.colors["coral"])
		style.configure("TNotebook", background=self.colors["paper"], borderwidth=0)
		style.configure("TNotebook.Tab", padding=(12, 7), font=("Avenir Next", 9, "bold"))
		style.map("TNotebook.Tab", foreground=[("selected", self.colors["teal"])])
		style.configure("Horizontal.TProgressbar", background=self.colors["teal"], troughcolor=self.colors["line"])

	def _create_layout(self) -> None:
		header = ttk.Frame(self.root, padding=(16, 12, 16, 8))
		header.pack(fill=tk.X)
		ttk.Label(header, text="AUVIZ", style="Title.TLabel").pack(side=tk.LEFT)
		ttk.Label(header, text="Wave Bundle Studio", style="Muted.TLabel").pack(side=tk.LEFT, padx=(12, 0), pady=(8, 0))
		save_project_button = ttk.Button(header, text="Save Project", command=self._save_project)
		save_project_button.pack(side=tk.RIGHT)
		open_project_button = ttk.Button(header, text="Open Project", command=self._open_project)
		open_project_button.pack(side=tk.RIGHT, padx=(0, 8))
		self.action_widgets.extend([open_project_button, save_project_button])

		body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
		body.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))
		controls_shell = ttk.Frame(body, width=390)
		plots_shell = ttk.Frame(body)
		body.add(controls_shell, weight=0)
		body.add(plots_shell, weight=1)

		self.control_notebook = ttk.Notebook(controls_shell)
		self.control_notebook.pack(fill=tk.BOTH, expand=True)
		self.audio_tab = ttk.Frame(self.control_notebook, padding=14)
		self.tone_tab = ttk.Frame(self.control_notebook, padding=14)
		self.color_tab = ttk.Frame(self.control_notebook, padding=14)
		self.export_tab = ttk.Frame(self.control_notebook, padding=14)
		self.control_notebook.add(self.audio_tab, text="Audio")
		self.control_notebook.add(self.tone_tab, text="Tone")
		self.control_notebook.add(self.color_tab, text="Color")
		self.control_notebook.add(self.export_tab, text="Export")
		self._build_audio_tab()
		self._build_tone_tab()
		self._build_color_tab()
		self._build_export_tab()

		self.plot_notebook = ttk.Notebook(plots_shell)
		self.plot_notebook.pack(fill=tk.BOTH, expand=True)
		frame_tab = ttk.Frame(self.plot_notebook)
		spectrum_tab = ttk.Frame(self.plot_notebook)
		self.timeline_tab = ttk.Frame(self.plot_notebook)
		self.plot_notebook.add(frame_tab, text="Frame")
		self.plot_notebook.add(spectrum_tab, text="Spectrum")
		self.plot_notebook.add(self.timeline_tab, text="Timeline")
		self._build_frame_view(frame_tab)
		self._build_spectrum_view(spectrum_tab)
		self._build_timeline_view(self.timeline_tab)

		status = ttk.Frame(self.root, padding=(14, 2, 14, 10))
		status.pack(fill=tk.X)
		self.progress_bar = ttk.Progressbar(status, mode="determinate", maximum=1.0)
		self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
		ttk.Label(status, textvariable=self.status_var, style="Muted.TLabel", width=42, anchor="e").pack(side=tk.RIGHT, padx=(12, 0))

	def _build_audio_tab(self) -> None:
		tab = self.audio_tab
		ttk.Label(tab, text="Source", style="Section.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
		ttk.Label(tab, textvariable=self.file_name_var, wraplength=330).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 2))
		ttk.Label(tab, textvariable=self.audio_info_var, style="Muted.TLabel", wraplength=330).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 10))
		choose_button = ttk.Button(tab, text="Choose Audio", command=self._choose_audio)
		choose_button.grid(row=3, column=0, sticky="ew", padx=(0, 5))
		analyze_button = ttk.Button(tab, text="Analyze", style="Accent.TButton", command=self._analyze)
		analyze_button.grid(row=3, column=1, sticky="ew", padx=(5, 0))
		self.action_widgets.extend([choose_button, analyze_button])

		ttk.Separator(tab).grid(row=4, column=0, columnspan=2, sticky="ew", pady=16)
		ttk.Label(tab, text="Analysis", style="Section.TLabel").grid(row=5, column=0, columnspan=2, sticky="w")
		self._add_entry(tab, 6, "Frames per second", self.fps_var)
		self._add_entry(tab, 7, "FFT window", self.fft_size_var)
		self._add_entry(tab, 8, "Low frequency (Hz)", self.low_frequency_var)
		self._add_entry(tab, 9, "High frequency (Hz)", self.high_frequency_var)
		tab.columnconfigure(0, weight=1)
		tab.columnconfigure(1, weight=1)

	def _build_tone_tab(self) -> None:
		tab = self.tone_tab
		ttk.Label(tab, text="Three-band EQ", style="Section.TLabel").pack(anchor="w")
		self.eq_editor = EQCurveEditor(tab, self._eq_changed)
		self.eq_editor.pack(fill=tk.X, pady=(6, 2))
		ttk.Button(tab, text="Reset EQ", command=self.eq_editor.reset).pack(anchor="e", pady=(0, 8))

		self._add_slider(tab, "Outlier threshold (IQR)", self.outlier_iqr_multiplier_var, 0.0, 6.0, "{:.2f}")
		self._add_slider(tab, "Small-coefficient clip", self.clip_fraction_var, 0.0, 0.99, "{:.2f}")
		self._add_slider(tab, "Contrast", self.contrast_var, -1.0, 1.0, "{:+.2f}")
		self._add_slider(tab, "Compression threshold", self.compression_threshold_var, 0.0, 1.0, "{:.2f}")
		self._add_slider(tab, "Compression amount", self.compression_amount_var, 0.0, 1.0, "{:.2f}")
		self._add_slider(tab, "Temporal smoothing", self.smoothing_var, 0.0, 1.0, "{:.2f}")
		apply_button = ttk.Button(tab, text="Apply Processing", style="Accent.TButton", command=self._process)
		apply_button.pack(fill=tk.X, pady=(12, 0))
		self.action_widgets.append(apply_button)

	def _build_color_tab(self) -> None:
		tab = self.color_tab
		ttk.Label(tab, text="Colormap", style="Section.TLabel").pack(anchor="w")
		self.colormap_picker = ColormapPicker(
			tab,
			self.colormap_var,
			self._colormap_names(),
			self._colormap_for_name,
			self._colormap_changed,
		)
		self.colormap_picker.pack(fill=tk.X, pady=(6, 5))
		self.colormap_swatch = tk.Canvas(tab, height=24, highlightthickness=1, highlightbackground=self.colors["line"])
		self.colormap_swatch.pack(fill=tk.X, pady=(0, 14))

		ttk.Label(tab, text="Custom palette", style="Section.TLabel").pack(anchor="w")
		self.palette_frame = ttk.Frame(tab)
		self.palette_frame.pack(fill=tk.X, pady=(7, 5))
		palette_actions = ttk.Frame(tab)
		palette_actions.pack(anchor="e")
		ttk.Button(palette_actions, text="+", width=3, command=self._add_color).pack(side=tk.LEFT, padx=2)
		ttk.Button(palette_actions, text="-", width=3, command=self._remove_color).pack(side=tk.LEFT, padx=2)

		name_row = ttk.Frame(tab)
		name_row.pack(fill=tk.X, pady=(12, 2))
		ttk.Label(name_row, text="Palette name").pack(side=tk.LEFT)
		ttk.Entry(name_row, textvariable=self.custom_colormap_name_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
		palette_storage = ttk.Frame(tab)
		palette_storage.pack(fill=tk.X, pady=(3, 8))
		ttk.Button(palette_storage, text="Delete Saved", command=self._delete_custom_colormap).pack(side=tk.LEFT)
		ttk.Button(palette_storage, text="Save Palette", style="Accent.TButton", command=self._save_custom_colormap).pack(side=tk.RIGHT)

		self._add_slider(tab, "Color upper bound", self.colormap_upper_var, 0.0, 1.0, "{:.2f}", self._color_control_changed)

	def _build_export_tab(self) -> None:
		tab = self.export_tab
		ttk.Label(tab, text="Output", style="Section.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
		ttk.Label(tab, text="Resolution").grid(row=1, column=0, sticky="w", pady=(10, 4))
		ttk.Combobox(
			tab,
			textvariable=self.resolution_var,
			values=("960 x 540", "1280 x 720", "1920 x 1080"),
			state="readonly",
			width=14,
		).grid(row=1, column=1, sticky="e", pady=(10, 4))
		ttk.Label(tab, text="Video path").grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 3))
		ttk.Entry(tab, textvariable=self.output_path_var).grid(row=3, column=0, sticky="ew", padx=(0, 5))
		browse_output = ttk.Button(tab, text="Browse", command=self._choose_output)
		browse_output.grid(row=3, column=1, sticky="ew")
		self.action_widgets.append(browse_output)

		ttk.Separator(tab).grid(row=4, column=0, columnspan=2, sticky="ew", pady=16)
		ttk.Label(tab, text="Render progress", style="Section.TLabel").grid(row=5, column=0, columnspan=2, sticky="w")
		self.export_progress_bar = ttk.Progressbar(tab, mode="determinate", maximum=100.0)
		self.export_progress_bar.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 3))
		ttk.Label(tab, textvariable=self.export_progress_var, style="Muted.TLabel").grid(
			row=7, column=0, columnspan=2, sticky="w", pady=(0, 14)
		)
		sample_button = ttk.Button(tab, textvariable=self.sample_button_var, style="Accent.TButton", command=self._render_sample)
		sample_button.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(0, 7))
		full_button = ttk.Button(tab, text="Render Full Video", command=self._render_full)
		full_button.grid(row=9, column=0, columnspan=2, sticky="ew", pady=7)
		open_button = ttk.Button(tab, text="Open Last Video", command=self._open_last_video)
		open_button.grid(row=10, column=0, columnspan=2, sticky="ew", pady=7)
		self.cancel_button = ttk.Button(tab, text="Cancel", style="Danger.TButton", command=self._cancel, state=tk.DISABLED)
		self.cancel_button.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(22, 0))
		self.action_widgets.extend([sample_button, full_button, open_button])
		tab.columnconfigure(0, weight=1)

	def _build_frame_view(self, parent: ttk.Frame) -> None:
		controls = ttk.Frame(parent, padding=(10, 8))
		controls.pack(fill=tk.X)
		ttk.Label(controls, text="Frame").pack(side=tk.LEFT)
		self.frame_scale = ttk.Scale(
			controls,
			variable=self.frame_var,
			from_=0,
			to=0,
			command=self._frame_slider_changed,
		)
		self.frame_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
		frame_entry = ttk.Entry(controls, textvariable=self.frame_entry_var, width=7)
		frame_entry.pack(side=tk.LEFT)
		frame_entry.bind("<Return>", self._frame_entry_changed)
		ttk.Label(controls, textvariable=self.frame_time_var, style="Muted.TLabel", width=11, anchor="e").pack(side=tk.LEFT, padx=(8, 0))

		self.frame_figure = Figure(figsize=(9.0, 5.2), dpi=100)
		self.frame_figure.patch.set_facecolor(self.colors["paper"])
		self.frame_axis = self.frame_figure.add_subplot(111)
		self.frame_axis.set_axis_off()
		self.frame_axis.imshow(np.zeros((180, 320, 3), dtype=np.uint8), origin="upper")
		self.frame_canvas = FigureCanvasTkAgg(self.frame_figure, master=parent)
		self.frame_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

	def _build_spectrum_view(self, parent: ttk.Frame) -> None:
		self.spectrum_figure = Figure(figsize=(8.8, 5.8), dpi=100)
		self.spectrum_figure.patch.set_facecolor(self.colors["paper"])
		self.spectrum_axes = self.spectrum_figure.subplots(
			3,
			1,
			gridspec_kw={"height_ratios": (1.0, 1.0, 1.15)},
		)
		self.spectrum_canvas = FigureCanvasTkAgg(self.spectrum_figure, master=parent)
		self.spectrum_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
		self._style_spectrum_axes()

	def _build_timeline_view(self, parent: ttk.Frame) -> None:
		parent.columnconfigure(0, weight=1)
		parent.rowconfigure(0, weight=1)

		preview = ttk.Frame(parent, padding=(10, 8, 10, 4))
		preview.grid(row=0, column=0, sticky="nsew")
		preview.columnconfigure(0, weight=1)
		preview.rowconfigure(0, weight=1)
		self.sample_figure = Figure(figsize=(8.8, 4.8), dpi=100)
		self.sample_figure.patch.set_facecolor(self.colors["paper"])
		self.sample_axis = self.sample_figure.add_subplot(111)
		self.sample_axis.set_facecolor("#10171c")
		self.sample_axis.set_axis_off()
		self.sample_image = self.sample_axis.imshow(
			np.zeros((self.PREVIEW_HEIGHT, self.PREVIEW_WIDTH, 3), dtype=np.uint8),
			origin="upper",
			interpolation="nearest",
		)
		self.sample_canvas = FigureCanvasTkAgg(self.sample_figure, master=preview)
		self.sample_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

		player_controls = ttk.Frame(preview, padding=(4, 5, 4, 0))
		player_controls.grid(row=1, column=0, sticky="ew")
		self.sample_play_button = ttk.Button(
			player_controls,
			text="Play",
			width=9,
			command=self._toggle_sample_playback,
			state=tk.DISABLED,
		)
		self.sample_play_button.pack(side=tk.LEFT)
		ttk.Label(
			player_controls,
			textvariable=self.sample_preview_var,
			style="Muted.TLabel",
		).pack(side=tk.LEFT, padx=(10, 0))

		waveform = ttk.Frame(parent, padding=(10, 2, 10, 0))
		waveform.grid(row=1, column=0, sticky="ew")
		self.timeline_figure = Figure(figsize=(9.0, 1.45), dpi=100)
		self.timeline_figure.patch.set_facecolor(self.colors["paper"])
		self.timeline_axis = self.timeline_figure.add_subplot(111)
		self.timeline_canvas = FigureCanvasTkAgg(self.timeline_figure, master=waveform)
		self.timeline_canvas.get_tk_widget().pack(fill=tk.X)
		self.timeline_canvas.mpl_connect("button_press_event", self._timeline_clicked)

		slider_row = ttk.Frame(parent, padding=(12, 8, 12, 14))
		slider_row.grid(row=2, column=0, sticky="ew")
		slider_row.columnconfigure(1, weight=1)
		ttk.Label(slider_row, text="Start").grid(row=0, column=0, sticky="w")
		self.sample_scale = ttk.Scale(
			slider_row,
			variable=self.sample_start_var,
			from_=0.0,
			to=0.0,
			command=self._sample_slider_changed,
		)
		self.sample_scale.grid(row=0, column=1, sticky="ew", padx=(10, 8))
		ttk.Label(slider_row, textvariable=self.sample_time_var, style="Muted.TLabel", width=17, anchor="e").grid(row=0, column=2)
		ttk.Label(slider_row, text="Length").grid(row=0, column=3, padx=(14, 5))
		duration_entry = ttk.Spinbox(
			slider_row,
			textvariable=self.sample_duration_var,
			from_=self.MIN_SAMPLE_DURATION,
			to=self.MAX_SAMPLE_DURATION,
			increment=0.25,
			width=6,
			command=self._sample_duration_changed,
		)
		duration_entry.grid(row=0, column=4)
		duration_entry.bind("<Return>", self._sample_duration_changed)
		duration_entry.bind("<FocusOut>", self._sample_duration_changed)
		ttk.Label(slider_row, text="s", style="Muted.TLabel").grid(row=0, column=5, padx=(3, 0))

	def _add_entry(self, parent: ttk.Frame, row: int, label: str, variable: tk.Variable) -> None:
		ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
		ttk.Entry(parent, textvariable=variable, width=14).grid(row=row, column=1, sticky="e", pady=5)

	def _add_slider(
		self,
		parent: ttk.Frame,
		label: str,
		variable: tk.DoubleVar,
		minimum: float,
		maximum: float,
		value_format: str,
		on_change: Callable[[], None] | None = None,
	) -> None:
		container = ttk.Frame(parent)
		container.pack(fill=tk.X, pady=5)
		value_label = ttk.Label(container, text=value_format.format(variable.get()), style="Muted.TLabel", width=7, anchor="e")
		ttk.Label(container, text=label).pack(side=tk.LEFT)
		value_label.pack(side=tk.RIGHT)

		def update_label(*_args: object) -> None:
			try:
				value_label.configure(text=value_format.format(float(variable.get())))
			except (ValueError, tk.TclError):
				pass

		variable.trace_add("write", update_label)

		def changed(raw_value: str) -> None:
			value = float(raw_value)
			variable.set(value)
			if on_change is None:
				self._mark_processing_dirty()
			else:
				on_change()

		ttk.Scale(parent, variable=variable, from_=minimum, to=maximum, command=changed).pack(fill=tk.X, pady=(0, 3))

	@staticmethod
	def _project_path_value(path: Path | None, project_path: Path) -> dict[str, object] | None:
		if path is None:
			return None
		try:
			value = os.path.relpath(path.expanduser().resolve(), project_path.parent)
			return {"value": value, "relative": True}
		except (OSError, ValueError):
			return {"value": str(path), "relative": False}

	@staticmethod
	def _resolve_project_path(value: object, project_path: Path) -> Path | None:
		if value is None:
			return None
		if not isinstance(value, dict) or not isinstance(value.get("value"), str):
			raise ValueError("The project contains an invalid file path.")
		path = Path(value["value"]).expanduser()
		if bool(value.get("relative", False)):
			path = project_path.parent / path
		return path.resolve()

	def _project_payload(self, project_path: Path) -> dict[str, object]:
		return {
			"format": self.PROJECT_FORMAT,
			"version": self.PROJECT_VERSION,
			"paths": {
				"audio": self._project_path_value(self.audio_path, project_path),
				"output": self._project_path_value(Path(self.output_path_var.get()), project_path),
				"last_video": self._project_path_value(self.last_video, project_path),
			},
			"analysis": {
				"fps": int(self.fps_var.get()),
				"fft_size": int(self.fft_size_var.get()),
				"low_frequency": float(self.low_frequency_var.get()),
				"high_frequency": float(self.high_frequency_var.get()),
			},
			"processing": {
				"eq": asdict(self.eq_editor.settings()),
				"outlier_iqr_multiplier": float(self.outlier_iqr_multiplier_var.get()),
				"clip_fraction": float(self.clip_fraction_var.get()),
				"contrast": float(self.contrast_var.get()),
				"compression_threshold": float(self.compression_threshold_var.get()),
				"compression_amount": float(self.compression_amount_var.get()),
				"smoothing": float(self.smoothing_var.get()),
			},
			"color": {
				"colormap": self.colormap_var.get(),
				"custom_name": self.custom_colormap_name_var.get(),
				"custom_colors": list(self.custom_colors),
				"saved_colormaps": self.saved_colormaps,
				"upper_bound": float(self.colormap_upper_var.get()),
			},
			"export": {"resolution": self.resolution_var.get()},
			"view": {
				"frame_index": float(self.frame_var.get()),
				"sample_start": float(self.sample_start_var.get()),
				"sample_duration": float(self.sample_duration_var.get()),
				"control_tab": self.control_notebook.index("current"),
				"plot_tab": self.plot_notebook.index("current"),
			},
		}

	def _save_project(self) -> None:
		initial_name = self.project_path.name if self.project_path is not None else "visualizer.auviz"
		selected = filedialog.asksaveasfilename(
			title="Save AUVIZ project",
			defaultextension=".auviz",
			filetypes=[("AUVIZ project", "*.auviz"), ("JSON", "*.json")],
			initialfile=initial_name,
		)
		if not selected:
			return
		project_path = Path(selected).expanduser().resolve()
		try:
			payload = self._project_payload(project_path)
			project_path.parent.mkdir(parents=True, exist_ok=True)
			temporary_path = project_path.with_suffix(project_path.suffix + ".tmp")
			temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
			temporary_path.replace(project_path)
		except (OSError, ValueError, tk.TclError) as error:
			messagebox.showerror("Could not save project", str(error))
			return
		self.project_path = project_path
		self._update_project_title()
		self.status_var.set(f"Project saved | {project_path.name}")

	def _open_project(self) -> None:
		selected = filedialog.askopenfilename(
			title="Open AUVIZ project",
			filetypes=[("AUVIZ project", "*.auviz *.json"), ("All files", "*.*")],
		)
		if selected:
			self._load_project(Path(selected))

	def _load_project(self, project_path: Path) -> None:
		project_path = project_path.expanduser().resolve()
		try:
			payload = json.loads(project_path.read_text(encoding="utf-8"))
			if not isinstance(payload, dict) or payload.get("format") != self.PROJECT_FORMAT:
				raise ValueError("This is not an AUVIZ project file.")
			version = int(payload.get("version", 0))
			if version < 1 or version > self.PROJECT_VERSION:
				raise ValueError(f"Unsupported AUVIZ project version: {version}.")
			self._apply_project_payload(payload, project_path)
		except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError, tk.TclError) as error:
			messagebox.showerror("Could not open project", str(error))
			return
		self.project_path = project_path
		self._update_project_title()
		if self.audio_path is None:
			self.status_var.set(f"Project loaded | {project_path.name}")
		elif not self.audio_path.exists():
			self.status_var.set("Project loaded | audio file missing")
			messagebox.showwarning(
				"Audio file not found",
				f"The project settings were restored, but the audio file could not be found:\n{self.audio_path}",
			)
		else:
			self.status_var.set(f"Project loaded | analyzing {self.audio_path.name}")
			self._analyze()

	def _apply_project_payload(self, payload: dict[str, object], project_path: Path) -> None:
		paths = payload.get("paths", {})
		analysis = payload.get("analysis", {})
		processing = payload.get("processing", {})
		color = payload.get("color", {})
		export = payload.get("export", {})
		view = payload.get("view", {})
		if not all(isinstance(section, dict) for section in (paths, analysis, processing, color, export, view)):
			raise ValueError("The project settings are malformed.")
		self._clear_media_state()

		self.audio_path = self._resolve_project_path(paths.get("audio"), project_path)
		output_path = self._resolve_project_path(paths.get("output"), project_path)
		self.last_video = self._resolve_project_path(paths.get("last_video"), project_path)
		if self.audio_path is not None:
			self.file_name_var.set(self.audio_path.name)
			self.audio_info_var.set(str(self.audio_path.parent))
		if output_path is not None:
			self.output_path_var.set(str(output_path))

		self.fps_var.set(int(analysis.get("fps", 24)))
		self.fft_size_var.set(int(analysis.get("fft_size", 12_000)))
		self.low_frequency_var.set(float(analysis.get("low_frequency", 30.0)))
		self.high_frequency_var.set(float(analysis.get("high_frequency", 12_000.0)))

		eq_data = processing.get("eq", {})
		if not isinstance(eq_data, dict):
			raise ValueError("The project EQ settings are malformed.")
		eq_defaults = asdict(EQSettings())
		eq_settings = EQSettings(**{
			key: float(eq_data.get(key, default)) for key, default in eq_defaults.items()
		})
		self.eq_editor.set_settings(eq_settings)
		self.outlier_iqr_multiplier_var.set(float(processing.get("outlier_iqr_multiplier", 1.5)))
		self.clip_fraction_var.set(float(processing.get("clip_fraction", 0.25)))
		self.contrast_var.set(float(processing.get("contrast", 0.0)))
		self.compression_threshold_var.set(float(processing.get("compression_threshold", 0.7)))
		self.compression_amount_var.set(float(processing.get("compression_amount", 0.5)))
		self.smoothing_var.set(float(processing.get("smoothing", 0.3)))

		custom_colors = color.get("custom_colors", self.custom_colors)
		if not (
			isinstance(custom_colors, list)
			and len(custom_colors) >= 2
			and all(isinstance(item, str) and matplotlib.colors.is_color_like(item) for item in custom_colors)
		):
			raise ValueError("The project custom palette is malformed.")
		project_colormaps = color.get("saved_colormaps", {})
		if not isinstance(project_colormaps, dict):
			raise ValueError("The project saved palettes are malformed.")
		for name, colors in project_colormaps.items():
			if (
				isinstance(name, str)
				and isinstance(colors, list)
				and len(colors) >= 2
				and all(isinstance(item, str) and matplotlib.colors.is_color_like(item) for item in colors)
			):
				self.saved_colormaps[name] = list(colors)
		self.custom_colors = list(custom_colors)
		colormap_name = str(color.get("colormap", "turbo"))
		if colormap_name not in self._colormap_names():
			colormap_name = "Custom"
		self.colormap_var.set(colormap_name)
		self.custom_colormap_name_var.set(str(color.get("custom_name", "")))
		self.colormap_upper_var.set(float(color.get("upper_bound", 0.3)))
		self.colormap_picker.set_choices(self._colormap_names())
		self._redraw_palette_buttons()
		self._redraw_colormap_swatch()

		self.resolution_var.set(str(export.get("resolution", "1280 x 720")))
		self.sample_duration_var.set(float(view.get("sample_duration", self.DEFAULT_SAMPLE_DURATION)))
		self.pending_project_view = {
			"frame_index": float(view.get("frame_index", 0.0)),
			"sample_start": float(view.get("sample_start", 0.0)),
		}
		self._sample_duration_changed()
		self.sample_start_var.set(self.pending_project_view["sample_start"])
		self._update_sample_span()
		self.control_notebook.select(int(np.clip(int(view.get("control_tab", 0)), 0, 3)))
		self.plot_notebook.select(int(np.clip(int(view.get("plot_tab", 0)), 0, 2)))
		self.processing_dirty = True

	def _clear_media_state(self) -> None:
		self._stop_sample_playback(reset=False)
		self.analysis = None
		self.processed = None
		self.preview_renderer = None
		self.normalization_reference = 1.0
		self.sample_preview_frames = None
		self.sample_preview_path = None
		self.sample_preview_fps = 0.0
		self.sample_preview_var.set("No sample rendered")
		self.sample_play_button.configure(state=tk.DISABLED, text="Play")
		self.frame_scale.configure(to=0)
		self.sample_scale.configure(to=0.0)
		self.frame_axis.clear()
		self.frame_axis.set_axis_off()
		self.frame_axis.imshow(np.zeros((180, 320, 3), dtype=np.uint8), origin="upper")
		self.frame_canvas.draw_idle()
		for axis in self.spectrum_axes:
			axis.clear()
		self._style_spectrum_axes()
		self.spectrum_canvas.draw_idle()
		self.timeline_axis.clear()
		self.sample_span = None
		self.timeline_canvas.draw_idle()

	def _update_project_title(self) -> None:
		name = self.project_path.name if self.project_path is not None else "Untitled"
		self.root.title(f"AUVIZ - {name}")

	def _choose_audio(self) -> None:
		selected = filedialog.askopenfilename(
			title="Choose audio",
			filetypes=[
				("Audio", "*.wav *.flac *.ogg *.aif *.aiff *.mp3 *.m4a"),
				("All files", "*.*"),
			],
		)
		if selected:
			self.audio_path = Path(selected)
			self.file_name_var.set(self.audio_path.name)
			self.audio_info_var.set(str(self.audio_path.parent))
			self.output_path_var.set(str(self.audio_path.with_name(f"{self.audio_path.stem}_visualizer.mp4")))
			self.status_var.set("Ready to analyze")

	def _analysis_settings(self) -> AnalysisSettings:
		settings = AnalysisSettings(
			fps=int(self.fps_var.get()),
			fft_size=int(self.fft_size_var.get()),
			low_frequency=float(self.low_frequency_var.get()),
			high_frequency=float(self.high_frequency_var.get()),
		)
		if settings.fps < 1 or settings.fps > 120:
			raise ValueError("Frames per second must be between 1 and 120.")
		if settings.fft_size < 256:
			raise ValueError("The FFT window must contain at least 256 samples.")
		return settings

	def _processing_settings(self) -> ProcessingSettings:
		return ProcessingSettings(
			eq=self.eq_editor.settings(),
			outlier_iqr_multiplier=float(self.outlier_iqr_multiplier_var.get()),
			clip_fraction=float(self.clip_fraction_var.get()),
			contrast=float(self.contrast_var.get()),
			compression_threshold=float(self.compression_threshold_var.get()),
			compression_amount=float(self.compression_amount_var.get()),
			smoothing=float(self.smoothing_var.get()),
		)

	def _analyze(self) -> None:
		if self.audio_path is None:
			self._choose_audio()
		if self.audio_path is None:
			return
		try:
			analysis_settings = self._analysis_settings()
			processing_settings = self._processing_settings()
		except (ValueError, tk.TclError) as error:
			messagebox.showerror("Invalid settings", str(error))
			return

		path = self.audio_path

		def task() -> tuple[AudioAnalysis, ProcessedAudio]:
			analysis = analyze_audio(path, analysis_settings, self._queue_progress, self.cancel_event)
			processed = process_audio(analysis, processing_settings, self._queue_progress, self.cancel_event)
			return analysis, processed

		self._submit(task, self._analysis_complete)

	def _analysis_complete(self, result: tuple[AudioAnalysis, ProcessedAudio]) -> None:
		self.analysis, processed = result
		self._processed_complete(processed)
		analysis = self.analysis
		self.file_name_var.set(analysis.source_path.name)
		self.audio_info_var.set(
			f"{analysis.duration:.2f} s | {analysis.sample_rate:,} Hz | "
			f"{analysis.frame_count:,} frames | {analysis.frequencies.size:,} bins"
		)
		self.frame_scale.configure(to=max(0, analysis.frame_count - 1))
		self._sample_duration_changed()
		if self.pending_project_view is None:
			frame_index = 0
			sample_start = 0.0
		else:
			frame_index = int(np.clip(
				round(self.pending_project_view["frame_index"]),
				0,
				analysis.frame_count - 1,
			))
			maximum_start = max(0.0, analysis.duration - self._sample_duration())
			sample_start = float(np.clip(
				self.pending_project_view["sample_start"],
				0.0,
				maximum_start,
			))
			self.pending_project_view = None
		self.frame_var.set(frame_index)
		self.frame_entry_var.set(str(frame_index))
		self.frame_time_var.set(f"{frame_index / analysis.fps:.2f} s")
		self.sample_start_var.set(sample_start)
		self._draw_waveform()
		self._render_current_frame()
		self.status_var.set("Analysis complete")

	def _process(self, after: Callable[[], None] | None = None) -> None:
		if self.analysis is None:
			self._analyze()
			return
		try:
			settings = self._processing_settings()
		except (ValueError, tk.TclError) as error:
			messagebox.showerror("Invalid settings", str(error))
			return
		analysis = self.analysis

		def task() -> ProcessedAudio:
			return process_audio(analysis, settings, self._queue_progress, self.cancel_event)

		def completed(processed: ProcessedAudio) -> None:
			self._processed_complete(processed)
			if after is not None:
				after()

		self._submit(task, completed)

	def _processed_complete(self, processed: ProcessedAudio) -> None:
		self.processed = processed
		self.processing_dirty = False
		self.preview_renderer = CartesianWaveRenderer(
			self.PREVIEW_WIDTH,
			self.PREVIEW_HEIGHT,
			processed.analysis.frequencies,
		)
		self.normalization_reference = max(
			float(np.max(np.sum(processed.spectra, axis=(1, 2)))),
			np.finfo(np.float32).eps,
		)
		outliers = [item.outlier_count for item in processed.state.channel_normalizations]
		self.status_var.set(f"Processing ready | outliers scaled: L {outliers[0]}, R {outliers[1]}")
		self._render_current_frame()

	def _eq_changed(self, _settings: EQSettings) -> None:
		self._mark_processing_dirty()

	def _mark_processing_dirty(self) -> None:
		if self.analysis is not None:
			self.processing_dirty = True
			self.status_var.set("Controls changed | processing update pending")

	def _color_control_changed(self) -> None:
		if self.processed is not None:
			self._schedule_preview(50)

	@staticmethod
	def _colormap_store_path() -> Path:
		override = os.environ.get("AUVIZ_CONFIG_DIR")
		if override:
			return Path(override).expanduser() / "colormaps.json"
		if sys.platform == "darwin":
			folder = Path.home() / "Library" / "Application Support" / "AUVIZ"
		elif os.name == "nt":
			folder = Path(os.environ.get("APPDATA", Path.home())) / "AUVIZ"
		else:
			folder = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "auviz"
		return folder / "colormaps.json"

	def _load_saved_colormaps(self) -> dict[str, list[str]]:
		try:
			payload = json.loads(self._colormap_store_path().read_text(encoding="utf-8"))
		except (OSError, json.JSONDecodeError):
			return {}
		if not isinstance(payload, dict):
			return {}
		palettes: dict[str, list[str]] = {}
		for name, colors in payload.items():
			if (
				isinstance(name, str)
				and isinstance(colors, list)
				and len(colors) >= 2
				and all(isinstance(color, str) and matplotlib.colors.is_color_like(color) for color in colors)
			):
				palettes[name] = list(colors)
		return palettes

	def _persist_saved_colormaps(self, palettes: dict[str, list[str]]) -> None:
		path = self._colormap_store_path()
		path.parent.mkdir(parents=True, exist_ok=True)
		temporary_path = path.with_suffix(".tmp")
		temporary_path.write_text(json.dumps(palettes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
		temporary_path.replace(path)

	def _colormap_names(self) -> list[str]:
		featured = ["turbo", "viridis", "inferno", "magma", "cividis", "RdBu_r"]
		return list(
			dict.fromkeys([*featured, "Custom", *sorted(self.saved_colormaps), *self.builtin_colormap_names])
		)

	def _colormap_for_name(self, name: str) -> Colormap:
		if name in self.saved_colormaps:
			return LinearSegmentedColormap.from_list(name, self.saved_colormaps[name], N=256)
		return build_colormap(name, self.custom_colors)

	def _colormap_changed(self, _event: object | None = None) -> None:
		name = self.colormap_var.get()
		if name in self.saved_colormaps:
			self.custom_colors = list(self.saved_colormaps[name])
			self.custom_colormap_name_var.set(name)
			self._redraw_palette_buttons()
		elif name != "Custom":
			self.custom_colormap_name_var.set("")
		self._redraw_colormap_swatch()
		self._color_control_changed()

	def _current_colormap(self) -> Colormap:
		return self._colormap_for_name(self.colormap_var.get())

	def _save_custom_colormap(self) -> None:
		name = self.custom_colormap_name_var.get().strip()
		if not name:
			messagebox.showerror("Palette name required", "Type a name before saving the palette.")
			return
		reserved_names = {item.casefold() for item in ["Custom", *self.builtin_colormap_names]}
		if name.casefold() in reserved_names:
			messagebox.showerror("Palette name unavailable", "Choose a name that is not a built-in colormap.")
			return
		updated = dict(self.saved_colormaps)
		updated[name] = list(self.custom_colors)
		try:
			self._persist_saved_colormaps(updated)
		except OSError as error:
			messagebox.showerror("Could not save palette", str(error))
			return
		self.saved_colormaps = updated
		self.colormap_var.set(name)
		self.colormap_picker.set_choices(self._colormap_names())
		self._redraw_colormap_swatch()
		self._color_control_changed()
		self.status_var.set(f"Saved colormap | {name}")

	def _delete_custom_colormap(self) -> None:
		name = self.custom_colormap_name_var.get().strip()
		if name not in self.saved_colormaps:
			messagebox.showinfo("No saved palette", "Select a saved custom palette to delete it.")
			return
		updated = dict(self.saved_colormaps)
		updated.pop(name)
		try:
			self._persist_saved_colormaps(updated)
		except OSError as error:
			messagebox.showerror("Could not delete palette", str(error))
			return
		self.saved_colormaps = updated
		self.colormap_var.set("Custom")
		self.custom_colormap_name_var.set("")
		self.colormap_picker.set_choices(self._colormap_names())
		self.colormap_picker.refresh_image("Custom")
		self._redraw_colormap_swatch()
		self._color_control_changed()
		self.status_var.set(f"Deleted colormap | {name}")

	def _redraw_colormap_swatch(self) -> None:
		if not hasattr(self, "colormap_swatch"):
			return
		self.colormap_swatch.delete("all")
		width = max(256, self.colormap_swatch.winfo_width())
		height = max(20, self.colormap_swatch.winfo_height())
		colors = colormap_lut(self._current_colormap())
		for index, color in enumerate(colors):
			x0 = index * width / 256.0
			x1 = (index + 1) * width / 256.0 + 1.0
			hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
			self.colormap_swatch.create_rectangle(x0, 0, x1, height, fill=hex_color, outline="")

	def _redraw_palette_buttons(self) -> None:
		for child in self.palette_frame.winfo_children():
			child.destroy()
		for index, color in enumerate(self.custom_colors):
			button = tk.Button(
				self.palette_frame,
				background=color,
				activebackground=color,
				width=3,
				height=1,
				relief=tk.FLAT,
				command=lambda color_index=index: self._choose_custom_color(color_index),
			)
			button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

	def _choose_custom_color(self, index: int) -> None:
		selected = colorchooser.askcolor(self.custom_colors[index], parent=self.root)[1]
		if selected:
			self.custom_colors[index] = selected
			self._custom_palette_changed()

	def _add_color(self) -> None:
		if len(self.custom_colors) >= 10:
			return
		selected = colorchooser.askcolor("#ffffff", parent=self.root)[1]
		if selected:
			self.custom_colors.append(selected)
			self._custom_palette_changed()

	def _remove_color(self) -> None:
		if len(self.custom_colors) <= 2:
			return
		self.custom_colors.pop()
		self._custom_palette_changed()

	def _custom_palette_changed(self) -> None:
		self.colormap_var.set("Custom")
		self.colormap_picker.refresh_image("Custom")
		self._redraw_palette_buttons()
		self._redraw_colormap_swatch()
		self._color_control_changed()

	def _frame_slider_changed(self, raw_value: str) -> None:
		frame_index = int(round(float(raw_value)))
		self.frame_entry_var.set(str(frame_index))
		if self.analysis is not None:
			self.frame_time_var.set(f"{frame_index / self.analysis.fps:.2f} s")
		self._schedule_preview()

	def _frame_entry_changed(self, _event: object | None = None) -> None:
		if self.analysis is None:
			return
		try:
			frame_index = int(self.frame_entry_var.get())
		except ValueError:
			frame_index = 0
		frame_index = int(np.clip(frame_index, 0, self.analysis.frame_count - 1))
		self.frame_var.set(frame_index)
		self.frame_entry_var.set(str(frame_index))
		self.frame_time_var.set(f"{frame_index / self.analysis.fps:.2f} s")
		self._schedule_preview(20)

	def _schedule_preview(self, delay: int = 120) -> None:
		if self.preview_after_id is not None:
			self.root.after_cancel(self.preview_after_id)
		self.preview_after_id = self.root.after(delay, self._render_current_frame)

	def _render_current_frame(self) -> None:
		self.preview_after_id = None
		if self.processed is None or self.preview_renderer is None:
			return
		frame_index = int(np.clip(round(self.frame_var.get()), 0, self.processed.analysis.frame_count - 1))
		try:
			rgb = render_frame_rgb(
				self.processed,
				frame_index,
				self.preview_renderer,
				self._current_colormap(),
				self.colormap_upper_var.get(),
				self.normalization_reference,
			)
		except Exception as error:
			self.status_var.set(str(error))
			return
		self.frame_axis.clear()
		self.frame_axis.imshow(rgb, origin="upper", interpolation="nearest")
		self.frame_axis.set_axis_off()
		self.frame_axis.set_title(
			f"Frame {frame_index} | {frame_index / self.processed.analysis.fps:.2f} s",
			color=self.colors["ink"],
			fontsize=11,
			pad=8,
		)
		self.frame_figure.tight_layout(pad=0.5)
		self.frame_canvas.draw_idle()
		self._draw_spectrum(frame_index)

	def _style_spectrum_axes(self) -> None:
		for axis in np.ravel(self.spectrum_axes):
			axis.set_facecolor(self.colors["paper"])
			axis.grid(True, which="both", color=self.colors["line"], linewidth=0.6, alpha=0.8)
			axis.tick_params(labelsize=8, colors=self.colors["muted"])
			for spine in axis.spines.values():
				spine.set_color(self.colors["line"])

	def _draw_spectrum(self, frame_index: int) -> None:
		if self.processed is None:
			return
		before, after = diagnostic_clipping_frame(self.processed, frame_index)
		frequencies = self.processed.analysis.frequencies
		channel_specs = (("Left", 0, self.colors["teal"]), ("Right", 1, self.colors["coral"]))
		for axis in self.spectrum_axes[:2]:
			axis.set_xscale("linear")
		for axis, (name, channel, color) in zip(self.spectrum_axes[:2], channel_specs):
			axis.clear()
			axis.plot(frequencies, before[:, channel], color="#8a9a9d", linewidth=0.8, alpha=0.8, label="Before")
			axis.plot(frequencies, after[:, channel], color=color, linewidth=1.0, label="After")
			threshold = self.clip_fraction_var.get() * float(np.max(before[:, channel]))
			axis.axhline(threshold, color=self.colors["gold"], linewidth=0.8, linestyle="--", label="Threshold")
			axis.set_xscale("log")
			axis.set_ylabel(f"{name} magnitude", fontsize=8)
			axis.legend(loc="upper right", fontsize=7, ncols=3, frameon=False)
		self.spectrum_axes[1].set_xlabel("Frequency (Hz)", fontsize=8)
		self.spectrum_axes[0].set_title(f"Small-coefficient clipping | frame {frame_index}", fontsize=10)

		maxima_axis = self.spectrum_axes[2]
		maxima_axis.clear()
		frame_indices = np.arange(self.processed.analysis.frame_count)
		for name, channel, color in channel_specs:
			normalization = self.processed.state.channel_normalizations[channel]
			raw_maxima = normalization.frame_maxima
			clipped_maxima = raw_maxima * normalization.frame_scales
			maxima_axis.plot(
				frame_indices,
				raw_maxima,
				color=color,
				linewidth=0.75,
				alpha=0.35,
				label=f"{name} raw",
			)
			maxima_axis.plot(
				frame_indices,
				clipped_maxima,
				color=color,
				linewidth=1.15,
				label=f"{name} clipped",
			)
			maxima_axis.axhline(
				normalization.high_threshold,
				color=color,
				linewidth=0.9,
				linestyle="--",
				label=f"{name} threshold",
			)
		maxima_axis.axvline(frame_index, color=self.colors["ink"], linewidth=0.8, alpha=0.7)
		maxima_axis.set_xlim(0, max(1, self.processed.analysis.frame_count - 1))
		maxima_axis.set_xlabel("Frame index", fontsize=8)
		maxima_axis.set_ylabel("Max FFT coefficient", fontsize=8)
		maxima_axis.set_title(
			f"Outlier clipping | IQR multiplier {self.processed.state.settings.outlier_iqr_multiplier:.2f}",
			fontsize=9,
		)
		maxima_axis.legend(loc="upper right", fontsize=6.5, ncols=3, frameon=False)
		self._style_spectrum_axes()
		self.spectrum_figure.tight_layout(pad=1.0)
		self.spectrum_canvas.draw_idle()

	def _draw_waveform(self) -> None:
		if self.analysis is None:
			return
		mono = np.mean(self.analysis.audio, axis=1)
		target_blocks = 1_800
		block_size = max(1, math.ceil(mono.size / target_blocks))
		padded_size = math.ceil(mono.size / block_size) * block_size
		padded = np.pad(mono, (0, padded_size - mono.size), constant_values=np.nan)
		blocks = padded.reshape(-1, block_size)
		low = np.nanmin(blocks, axis=1)
		high = np.nanmax(blocks, axis=1)
		times = np.arange(blocks.shape[0]) * block_size / self.analysis.sample_rate

		self.timeline_axis.clear()
		self.sample_span = None
		self.timeline_axis.fill_between(times, low, high, color=self.colors["teal"], alpha=0.72, linewidth=0)
		self.timeline_axis.axhline(0.0, color=self.colors["line"], linewidth=0.7)
		self.timeline_axis.set_xlim(0.0, self.analysis.duration)
		self.timeline_axis.set_ylim(-1.05, 1.05)
		self.timeline_axis.set_xlabel("Time (s)")
		self.timeline_axis.set_yticks([])
		self.timeline_axis.set_facecolor(self.colors["paper"])
		for spine in self.timeline_axis.spines.values():
			spine.set_color(self.colors["line"])
		self._update_sample_span()
		self.timeline_figure.tight_layout(pad=1.0)
		self.timeline_canvas.draw_idle()

	def _sample_slider_changed(self, raw_value: str) -> None:
		self.sample_start_var.set(float(raw_value))
		self._update_sample_span()

	def _sample_duration(self) -> float:
		try:
			duration = float(self.sample_duration_var.get())
		except (ValueError, tk.TclError):
			duration = self.DEFAULT_SAMPLE_DURATION
		duration = float(np.clip(duration, self.MIN_SAMPLE_DURATION, self.MAX_SAMPLE_DURATION))
		if self.analysis is not None:
			duration = min(duration, self.analysis.duration)
		return duration

	def _sample_duration_changed(self, _event: object | None = None) -> None:
		duration = self._sample_duration()
		self.sample_duration_var.set(round(duration, 2))
		self.sample_button_var.set(f"Render {duration:g} s Sample")
		if self.analysis is not None:
			maximum = max(0.0, self.analysis.duration - duration)
			self.sample_scale.configure(to=maximum)
			self.sample_start_var.set(min(float(self.sample_start_var.get()), maximum))
		self._update_sample_span()

	def _timeline_clicked(self, event: object) -> None:
		if self.analysis is None or getattr(event, "inaxes", None) is not self.timeline_axis:
			return
		x_value = getattr(event, "xdata", None)
		if x_value is None:
			return
		duration = self._sample_duration()
		maximum = max(0.0, self.analysis.duration - duration)
		start = float(np.clip(x_value - duration / 2.0, 0.0, maximum))
		self.sample_start_var.set(start)
		self._update_sample_span()

	def _update_sample_span(self) -> None:
		start = float(self.sample_start_var.get())
		end = start + self._sample_duration()
		self.sample_time_var.set(f"{start:.2f} - {end:.2f} s")
		if not hasattr(self, "timeline_axis"):
			return
		if self.sample_span is not None:
			self.sample_span.remove()
		self.sample_span = self.timeline_axis.axvspan(start, end, color=self.colors["coral"], alpha=0.24)
		self.timeline_canvas.draw_idle()

	def _choose_output(self) -> None:
		selected = filedialog.asksaveasfilename(
			title="Save visualization",
			defaultextension=".mp4",
			filetypes=[("MP4 video", "*.mp4")],
			initialfile=Path(self.output_path_var.get()).name,
		)
		if selected:
			self.output_path_var.set(selected)

	def _parse_resolution(self) -> tuple[int, int]:
		parts = self.resolution_var.get().lower().replace(" ", "").split("x")
		if len(parts) != 2:
			raise ValueError("Invalid output resolution.")
		return int(parts[0]), int(parts[1])

	def _render_sample(self) -> None:
		self.plot_notebook.select(self.timeline_tab)
		self.sample_preview_var.set(f"Rendering {self._sample_duration():g} s sample...")
		self._ensure_processed(lambda: self._start_export(sample=True))

	def _render_full(self) -> None:
		self._ensure_processed(lambda: self._start_export(sample=False))

	def _ensure_processed(self, action: Callable[[], None]) -> None:
		if self.analysis is None:
			self._analyze()
			return
		if self.processing_dirty or self.processed is None:
			self._process(after=action)
		else:
			action()

	def _start_export(self, sample: bool) -> None:
		if self.processed is None:
			return
		try:
			width, height = self._parse_resolution()
			output = Path(self.output_path_var.get()).expanduser()
			if output.suffix.lower() != ".mp4":
				output = output.with_suffix(".mp4")
			if sample:
				output = output.with_name(f"{output.stem}_sample.mp4")
			colormap = self._current_colormap()
		except (ValueError, KeyError) as error:
			messagebox.showerror("Invalid export settings", str(error))
			return

		processed = self.processed
		start_seconds = float(self.sample_start_var.get()) if sample else 0.0
		duration_seconds = self._sample_duration() if sample else None
		colormap_upper = float(self.colormap_upper_var.get())
		self.export_active = True
		self.export_progress_bar.configure(value=0.0)
		self.export_progress_var.set(
			f"Starting {duration_seconds:g} s sample..." if sample and duration_seconds is not None else "Starting full render..."
		)

		def task() -> ExportResult:
			result_path = export_video(
				processed,
				output,
				width,
				height,
				colormap,
				colormap_upper,
				start_seconds=start_seconds,
				duration_seconds=duration_seconds,
				progress=self._queue_progress,
				cancel_event=self.cancel_event,
			)
			if not sample:
				return ExportResult(path=result_path)
			self._queue_progress("Preparing preview", 0, 1)
			frames = decode_video_preview(
				result_path,
				self.PREVIEW_WIDTH,
				self.PREVIEW_HEIGHT,
				self.cancel_event,
			)
			self._queue_progress("Preparing preview", 1, 1)
			return ExportResult(
				path=result_path,
				preview_frames=frames,
				preview_fps=float(processed.analysis.fps),
			)

		self._submit(task, self._export_complete)

	def _export_complete(self, result: ExportResult) -> None:
		self.last_video = result.path
		self.status_var.set(f"Saved {result.path.name}")
		self.export_progress_bar.configure(value=100.0)
		self.export_progress_var.set(f"Complete | {result.path.name}")
		if result.preview_frames is None:
			messagebox.showinfo("Video complete", f"Saved to:\n{result.path}")
			return
		self.sample_preview_frames = result.preview_frames
		self.sample_preview_path = result.path
		self.sample_preview_fps = result.preview_fps
		self.sample_preview_var.set(f"{result.path.name} | {len(result.preview_frames) / result.preview_fps:.1f} s")
		self.sample_play_button.configure(state=tk.NORMAL, text="Play")
		self.plot_notebook.select(self.timeline_tab)
		self._start_sample_playback()

	def _toggle_sample_playback(self) -> None:
		if self.sample_playing:
			self._stop_sample_playback(reset=False)
		else:
			self._start_sample_playback()

	def _start_sample_playback(self) -> None:
		if self.sample_preview_frames is None or self.sample_preview_path is None:
			return
		self._stop_sample_playback(reset=True)
		self.sample_playing = True
		self.sample_playback_started = time.monotonic()
		self.sample_play_button.configure(text="Stop")
		if sys.platform == "darwin" and Path("/usr/bin/afplay").exists():
			self.sample_audio_process = subprocess.Popen(
				["/usr/bin/afplay", str(self.sample_preview_path)],
				stdout=subprocess.DEVNULL,
				stderr=subprocess.DEVNULL,
			)
		self._advance_sample_playback()

	def _advance_sample_playback(self) -> None:
		if not self.sample_playing or self.sample_preview_frames is None:
			return
		elapsed = time.monotonic() - self.sample_playback_started
		frame_index = min(
			int(elapsed * self.sample_preview_fps),
			len(self.sample_preview_frames) - 1,
		)
		self.sample_image.set_data(self.sample_preview_frames[frame_index])
		self.sample_canvas.draw_idle()
		if frame_index >= len(self.sample_preview_frames) - 1:
			self._stop_sample_playback(reset=False)
			self.sample_play_button.configure(text="Replay")
			return
		self.sample_playback_after_id = self.root.after(15, self._advance_sample_playback)

	def _stop_sample_playback(self, reset: bool) -> None:
		self.sample_playing = False
		if self.sample_playback_after_id is not None:
			self.root.after_cancel(self.sample_playback_after_id)
			self.sample_playback_after_id = None
		if self.sample_audio_process is not None:
			if self.sample_audio_process.poll() is None:
				self.sample_audio_process.terminate()
			self.sample_audio_process = None
		if reset and self.sample_preview_frames is not None:
			self.sample_image.set_data(self.sample_preview_frames[0])
			self.sample_canvas.draw_idle()
		if self.sample_preview_frames is not None:
			self.sample_play_button.configure(text="Replay")

	def _open_last_video(self) -> None:
		if self.last_video is None or not self.last_video.exists():
			messagebox.showinfo("No video", "Render a sample or full video first.")
			return
		if sys.platform == "darwin":
			subprocess.Popen(["open", str(self.last_video)])
		elif os.name == "nt":
			os.startfile(self.last_video)  # type: ignore[attr-defined]
		else:
			subprocess.Popen(["xdg-open", str(self.last_video)])

	def _queue_progress(self, phase: str, current: int, total: int) -> None:
		self.events.put(("progress", phase, current, total))

	def _submit(self, task: Callable[[], object], completed: Callable[[object], None]) -> None:
		if self.busy:
			return
		self.busy = True
		self.cancel_event.clear()
		self._set_actions_enabled(False)
		self.progress_bar.configure(value=0.0, maximum=1.0)
		future = self.executor.submit(task)
		future.add_done_callback(lambda finished: self.events.put(("done", finished, completed)))

	def _poll_events(self) -> None:
		try:
			while True:
				event = self.events.get_nowait()
				if event[0] == "progress":
					_, phase, current, total = event
					total_value = max(1, int(total))
					self.progress_bar.configure(maximum=total_value, value=int(current))
					self.status_var.set(f"{phase} | {int(current):,} / {total_value:,}")
					if getattr(self, "export_active", False):
						fraction = float(current) / total_value
						if phase == "Preparing preview":
							fraction = 1.0
						self.export_progress_bar.configure(value=100.0 * fraction)
						self.export_progress_var.set(f"{phase} | {fraction:.0%}")
				elif event[0] == "done":
					_, future, completed = event
					was_export = getattr(self, "export_active", False)
					self.busy = False
					self._set_actions_enabled(True)
					try:
						result = future.result()
					except WorkCancelled:
						self.status_var.set("Cancelled")
						if was_export:
							self.export_progress_var.set("Render cancelled")
					except Exception as error:
						self.status_var.set("Operation failed")
						if was_export:
							self.export_progress_var.set("Render failed")
						messagebox.showerror("AUVIZ", str(error))
					else:
						completed(result)
					finally:
						if was_export:
							self.export_active = False
		except queue.Empty:
			pass
		if self.root.winfo_exists():
			self.poll_after_id = self.root.after(80, self._poll_events)

	def _set_actions_enabled(self, enabled: bool) -> None:
		state = tk.NORMAL if enabled else tk.DISABLED
		for widget in self.action_widgets:
			widget.configure(state=state)
		self.cancel_button.configure(state=tk.DISABLED if enabled else tk.NORMAL)

	def _cancel(self) -> None:
		self.cancel_event.set()
		self.status_var.set("Cancelling")
		if self.export_active:
			self.export_progress_var.set("Cancelling render...")

	def _close(self) -> None:
		self.cancel_event.set()
		self._stop_sample_playback(reset=False)
		if self.preview_after_id is not None:
			self.root.after_cancel(self.preview_after_id)
			self.preview_after_id = None
		if self.poll_after_id is not None:
			self.root.after_cancel(self.poll_after_id)
			self.poll_after_id = None
		self.executor.shutdown(wait=False, cancel_futures=True)
		self.root.destroy()


def main() -> None:
	root = tk.Tk()
	AudioVisualizerApp(root)
	root.mainloop()


if __name__ == "__main__":
	main()
