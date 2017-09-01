SUBDIRS = . base common_audio system_wrappers modules

noinst_HEADERS = common.h \
		 common_types.h \
		 typedefs.h

noinst_LTLIBRARIES = libwebrtc.la

libwebrtc_la_SOURCES = common_types.cc
libwebrtc_la_CXXFLAGS = $(AM_CXXFLAGS) $(COMMON_CXXFLAGS)

EXTRA_DIST = BUILD.gn PATENTS LICENSE_THIRD_PARTY
