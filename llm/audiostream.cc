/*
 *  audiostream.cc - Raw PCM feed for the live A/V stream.
 *
 *  Copyright (C) 2026  The Exult Team
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 */

#ifdef HAVE_CONFIG_H
#	include <config.h>
#endif

#ifdef USE_LLM_AGENT

#include "audiostream.h"

#include <iostream>
#include <cstring>
#include <string>

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
	int    g_fd      = -1;
	bool   g_enabled = false;
	int    g_rate    = 48000;
	int    g_channels = 2;
	string g_path;
}    // namespace

namespace LLM_agent {

	bool AudioStream_init(const string& fifo_path, int rate, int channels) {
#ifdef _WIN32
		(void)fifo_path;
		(void)rate;
		(void)channels;
		return false;
#else
		g_rate     = rate > 0 ? rate : 48000;
		g_channels = channels > 0 ? channels : 2;
		g_path     = fifo_path;
		if (mkfifo(fifo_path.c_str(), 0600) != 0 && errno != EEXIST) {
			cerr << "LLM audio stream: mkfifo(" << fifo_path
				 << ") failed: " << strerror(errno) << endl;
			return false;
		}
		g_fd      = ::open(fifo_path.c_str(), O_WRONLY | O_NONBLOCK);
		g_enabled = true;    // retry the open in _write until a reader attaches
		cerr << "LLM audio stream: raw s16le PCM FIFO at " << fifo_path << " ("
			 << g_rate << " Hz, " << g_channels << " ch)" << endl;
		return true;
#endif
	}

	void AudioStream_write(const int16_t* pcm, uint32_t bytes) {
#ifndef _WIN32
		if (!g_enabled || !pcm || bytes == 0) {
			return;
		}
		// Lazily (re)open until a reader (ffmpeg) attaches.
		if (g_fd < 0) {
			g_fd = ::open(g_path.c_str(), O_WRONLY | O_NONBLOCK);
			if (g_fd < 0) {
				return;    // no reader yet
			}
#	ifdef F_SETPIPE_SZ
			// Large pipe (try 8MB, ~40s of 48k stereo) so encoder jitter can't
			// fill it and force dropped audio blocks. Kernel may cap this.
			fcntl(g_fd, F_SETPIPE_SZ, 8 * 1024 * 1024);
#	endif
		}
		// Write the whole block. This runs on the audio callback thread, so we
		// can't block indefinitely, but audio tolerates a short wait far better
		// than a dropped block (which is an audible click/cutoff). Retry for up
		// to ~40ms on a full pipe before giving up on this block.
		const auto*  buf = reinterpret_cast<const unsigned char*>(pcm);
		size_t       off = 0;
		int          tries = 0;
		while (off < bytes) {
			const ssize_t n = ::write(
					g_fd, buf + off, static_cast<size_t>(bytes) - off);
			if (n > 0) {
				off += static_cast<size_t>(n);
			} else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
				if (++tries > 8) {
					break;    // reader far behind; drop rest of this block
				}
				struct pollfd pfd{g_fd, POLLOUT, 0};
				poll(&pfd, 1, 2);
			} else {
				::close(g_fd);
				g_fd = -1;
				return;    // reader gone; reopen later
			}
		}
#else
		(void)pcm;
		(void)bytes;
#endif
	}

	void AudioStream_close() {
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

	bool AudioStream_running() {
		return g_enabled;
	}

}    // namespace LLM_agent

#endif /* USE_LLM_AGENT */
