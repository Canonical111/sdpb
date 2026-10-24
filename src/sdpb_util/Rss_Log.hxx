#pragma once

// lab (bfss #44): optional per-rank RSS log at timer boundaries.
// Enabled only if environment variable SDPB_RSS_LOG names an existing directory;
// each MPI rank then appends "seconds event name VmRSS_kB VmHWM_kB" lines to
// $SDPB_RSS_LOG/rss-rank<world rank>.tsv. No effect when the variable is unset.

#include <string>

void rss_log(const char *event, const std::string &name);
