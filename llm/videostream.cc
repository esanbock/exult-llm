/*
 *  videostream.cc - Raw frame feed for the live A/V stream.
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

#ifdef HAVE_CONFIG_H
#	include <config.h>
#endif

#ifdef USE_LLM_AGENT

#include "videostream.h"

#include "gamewin.h"
#include "imagewin.h"
#include "ignore_unused_variable_warning.h"

#include <SDL3/SDL.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#ifndef _WIN32
#	include <cerrno>
#	include <fcntl.h>
#	include <poll.h>
#	include <sys/stat.h>
#	include <sys/types.h>
#	include <unistd.h>
#endif

using std::cerr;
using std::endl;
using std::string;

namespace {

	int    g_fd        = -1;       // FIFO write fd (non-blocking).
	bool   g_enabled   = false;
	int    g_fps       = 15;
	uint32 g_last_ms   = 0;
	int    g_w         = 0;
	int    g_h         = 0;
	string g_path;
	std::vector<unsigned char> g_buf;

}    // namespace

namespace LLM_agent {

	bool Stream_init(const string& fifo_path, int target_fps) {
#ifdef _WIN32
		(void)fifo_path;
		(void)target_fps;
		cerr << "LLM stream: FIFO video not supported on Windows" << endl;
		return false;
#else
		g_fps  = target_fps > 0 ? target_fps : 15;
		g_path = fifo_path;
		// Create the FIFO if it doesn't already exist.
		if (mkfifo(fifo_path.c_str(), 0600) != 0 && errno != EEXIST) {
			cerr << "LLM stream: mkfifo(" << fifo_path << ") failed: "
				 << strerror(errno) << endl;
			return false;
		}
		// Open non-blocking for writing. O_NONBLOCK means open() returns even
		// with no reader yet (it would otherwise block). We (re)try opening in
		// Stream_frame() until a reader (ffmpeg) attaches.
		g_fd = ::open(fifo_path.c_str(), O_WRONLY | O_NONBLOCK);
		g_enabled = true;    // enabled even if no reader yet; we retry the open
		cerr << "LLM stream: raw RGB24 video FIFO at " << fifo_path
			 << " (target " << g_fps << " fps)" << endl;
		return true;
#endif
	}

	void Stream_frame() {
#ifndef _WIN32
		if (!g_enabled) {
			return;
		}
		// Rate-limit.
		const uint32 now = SDL_GetTicks();
		const uint32 interval = static_cast<uint32>(1000 / g_fps);
		if (g_last_ms != 0 && now - g_last_ms < interval) {
			return;
		}

		// Lazily (re)open the FIFO until a reader attaches.
		if (g_fd < 0) {
			g_fd = ::open(g_path.c_str(), O_WRONLY | O_NONBLOCK);
			if (g_fd < 0) {
				return;    // no reader yet
			}
			// Enlarge the pipe buffer to hold a few frames so a briefly-behind
			// encoder doesn't cause EAGAIN mid-frame. Best effort.
#ifdef F_SETPIPE_SZ
			fcntl(g_fd, F_SETPIPE_SZ, 8 * 1024 * 1024);
#endif
		}

		Game_window* gwin = Game_window::get_instance();
		if (!gwin || !gwin->get_win()) {
			return;
		}
		int w = 0;
		int h = 0;
		uint32_t pixfmt = 0;
		if (!gwin->get_win()->capture_rgb(g_buf, w, h, pixfmt)) {
			return;
		}
		g_w = w;
		g_h = h;
		g_last_ms = now;
		static bool logged_dims = false;
		if (!logged_dims) {
			logged_dims = true;
			// Map the SDL pixel format to the ffmpeg -pixel_format name so the
			// external encoder reads the native bytes with no conversion.
			const char* ff = "rgba";
			switch (static_cast<SDL_PixelFormat>(pixfmt)) {
			case SDL_PIXELFORMAT_ARGB8888: ff = "bgra"; break;  // little-endian byte order
			case SDL_PIXELFORMAT_XRGB8888: ff = "bgr0"; break;
			case SDL_PIXELFORMAT_RGBA8888: ff = "abgr"; break;
			case SDL_PIXELFORMAT_RGBX8888: ff = "0bgr"; break;
			case SDL_PIXELFORMAT_ABGR8888: ff = "rgba"; break;
			case SDL_PIXELFORMAT_XBGR8888: ff = "rgb0"; break;
			case SDL_PIXELFORMAT_BGRA8888: ff = "argb"; break;
			case SDL_PIXELFORMAT_BGRX8888: ff = "0rgb"; break;
			default: ff = "rgba"; break;
			}
			std::cerr << "LLM stream: first frame " << w << "x" << h
					  << " ffpixfmt=" << ff
					  << " sdlfmt=0x" << std::hex << pixfmt << std::dec
					  << " (" << g_buf.size()
					  << " bytes)" << std::endl;
		}

		// Write the ENTIRE frame, always aligned. The rawvideo reader (ffmpeg)
		// requires exactly w*h*4 bytes per frame with no gaps or partials - any
		// misalignment corrupts every subsequent frame. So we write the whole
		// frame, waiting (via poll) when the pipe is briefly full. Because the
		// encoder now drains continuously (HLS), these waits are short and the
		// game loop is not meaningfully stalled. On a genuine reader-gone error
		// we close so a new reader can reattach on a clean frame boundary.
		size_t off = 0;
		while (off < g_buf.size()) {
			const ssize_t n = ::write(g_fd, g_buf.data() + off, g_buf.size() - off);
			if (n > 0) {
				off += static_cast<size_t>(n);
			} else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
				struct pollfd pfd{g_fd, POLLOUT, 0};
				if (poll(&pfd, 1, 500) <= 0) {
					// Reader stalled >500ms: drop connection, resync next reader.
					::close(g_fd);
					g_fd = -1;
					return;
				}
			} else {
				::close(g_fd);
				g_fd = -1;
				return;
			}
		}
#endif
	}

	void Stream_close() {
#ifndef _WIN32
		if (g_fd >= 0) {
			::close(g_fd);
			g_fd = -1;
		}
		if (!g_path.empty()) {
			::unlink(g_path.c_str());
		}
		g_enabled = false;
#endif
	}

	bool Stream_running() {
		return g_enabled;
	}

	int Stream_width() {
		return g_w;
	}

	int Stream_height() {
		return g_h;
	}

}    // namespace LLM_agent

#endif /* USE_LLM_AGENT */
