/*
 *  audiostream.h - Raw PCM feed for the live A/V stream.
 *
 *  Copyright (C) 2026  The Exult Team
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 */

#ifndef INCL_LLM_AUDIOSTREAM_H
#define INCL_LLM_AUDIOSTREAM_H 1

#ifdef USE_LLM_AGENT

#include <cstdint>
#include <string>

/*
 *  Tap the mixed audio and write it to a FIFO as raw interleaved PCM, exactly
 *  mirroring the video FIFO. This bypasses the snd-aloop loopback entirely -
 *  the PCM is written from the same audio callback that produces it, so it is
 *  perfectly paced (no capture-clock starvation / fragmentation). An external
 *  ffmpeg reads it as "-f s16le -ar <rate> -ac <channels>".
 */
namespace LLM_agent {

	// Open the PCM FIFO. rate/channels describe the samples that will be fed
	// (so the reader can be configured to match). Non-blocking; drops audio if
	// no reader is attached. Safe to call once at startup.
	bool AudioStream_init(
			const std::string& fifo_path, int rate, int channels);

	// Write one block of interleaved 16-bit PCM (called from the audio mixer).
	// 'bytes' is the size of the block. Cheap no-op if streaming is disabled or
	// no reader attached.
	void AudioStream_write(const int16_t* pcm, uint32_t bytes);

	// Close and remove the FIFO.
	void AudioStream_close();

	bool AudioStream_running();

}    // namespace LLM_agent

#endif /* USE_LLM_AGENT */

#endif /* INCL_LLM_AUDIOSTREAM_H */
