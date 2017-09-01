noinst_LTLIBRARIES = libbase.la

noinst_HEADERS = arraysize.h \
		 atomicops.h \
		 basictypes.h \
		 constructormagic.h \
		 safe_conversions.h \
		 safe_conversions_impl.h \
		 scoped_ptr.h \
		 template_util.h \
		 thread_annotations.h

libbase_la_SOURCES = criticalsection.cc \
		     criticalsection.h \
		     checks.cc \
		     checks.h \
		     event.cc \
		     event.h \
		     platform_thread.cc \
		     platform_thread.h \
		     platform_file.cc \
		     platform_file.h \
		     stringutils.cc \
		     stringutils.h \
		     thread_checker.h \
		     thread_checker_impl.cc \
		     thread_checker_impl.h

libbase_la_CXXFLAGS = $(AM_CXXFLAGS) $(COMMON_CXXFLAGS)

EXTRA_DIST = BUILD.gn
