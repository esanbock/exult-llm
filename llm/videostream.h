/*
 *  videostream.h - Raw frame feed for the live A/V stream.
 *
 *  Copyright (C) 2026  The Exult Team
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 */

#ifndef INCL_LLM_VIDEOSTREAM_H
#define INCL_LLM_VIDEOSTREAM_H 1

#ifdef USE_LLM_AGENT

#include <string>

/*
 *  A minimal, dependency-free raw-video feed. Exult is the only process that
 *  can read its own SDL renderer, so the frame pull must live in-engine. This
 *  writes tightly-packed RGB24 frames to a FIFO (named pipe) that an external
 *  ffmpeg reads as "-f rawvideo -pix_fmt rgb24". ffmpeg muxes it with the ALSA
 *  loopback audio and streams to VLC/Twitch. Writing is non-blocking: if no
 *  reader is attached, frames are dropped so the game never stalls.
 *
 *  Call Stream_init(fifo_path) once after the window is up, Stream_frame() once
 *  per rendered frame (rate-limited internally), and Stream_close() at shutdown.
 */
namespace LLM_agent {

	// Open the FIFO for non-blocking writes at the given path (created if
	// missing). target_fps caps how often frames are actually emitted.
	bool Stream_init(const std::string& fifo_path, int target_fps = 15);

	// Grab the current frame and write it if a reader is attached and enough
	// time has elapsed since the last emitted frame. Cheap no-op otherwise.
	void Stream_frame();

	// Close the FIFO and release resources.
	void Stream_close();

	// True if streaming is enabled (Stream_init succeeded).
	bool Stream_running();

	// The frame geometry actually being emitted (0 until first frame).
	int Stream_width();
	int Stream_height();

}    // namespace LLM_agent

#endif /* USE_LLM_AGENT */

#endif /* INCL_LLM_VIDEOSTREAM_H */
