AC_INIT([webrtc-audio-processing], [0.3])
AM_INIT_AUTOMAKE([dist-xz subdir-objects tar-ustar])

AC_SUBST(LIBWEBRTC_AUDIO_PROCESSING_VERSION_INFO, [1:0:0])

AM_SILENT_RULES([yes])

# Set up the host_* variables
AC_CANONICAL_HOST

AC_PROG_CC
AC_PROG_CXX
AC_PROG_LIBTOOL
AC_PROG_INSTALL
AM_PROG_AS

AC_LANG_C
AC_LANG_CPLUSPLUS

AC_ARG_WITH([ns-mode],
	    AS_HELP_STRING([--with-ns-mode=float|fixed], [Noise suppresion mode to use. Default is float]))

AS_CASE(["${with_ns_mode}"],
	["fixed"], [NS_FIXED=1],
	["float"], [NS_FIXED=0],
	[NS_FIXED=0])
AM_CONDITIONAL(NS_FIXED, [test "x${NS_FIXED}" = "x1"])

AC_ARG_WITH(
  gnustl,
  AC_HELP_STRING(
    [--with-gnustl],
    [use gnustl @<:@default=no@:>@]),
  [AS_CASE(
    [$withval], [no], [], [yes], [],
    [AC_MSG_ERROR([bad value "$withval" for --with-gnustl])])],
  [with_gnustl=no])
if test "x$with_gnustl" != "xno"; then
  PKG_CHECK_MODULES(GNUSTL, gnustl)
fi
AC_SUBST(GNUSTL_LIBS)
AC_SUBST(GNUSTL_CFLAGS)


# Borrowed from gst-plugins-bad
AC_CHECK_HEADER(MobileCoreServices/MobileCoreServices.h, HAVE_IOS="yes", HAVE_IOS="no", [-])

# Based on gst-plugins-bad configure.ac and defines in
# <chromium source>/build/config/BUILDCONFIG.gn and
# webrtc/BUILD.gn
AS_CASE(["${host}"],
    [*android*],
        [
         OS_CFLAGS="-DWEBRTC_ANDROID -DWEBRTC_LINUX -DWEBRTC_THREAD_RR -DWEBRTC_CLOCK_TYPE_REALTIME"
         OS_LDFLAGS="-llog"
         PLATFORM_CFLAGS="-DWEBRTC_POSIX"
         HAVE_POSIX=1
        ],
    [*-*linux*],
        [
         OS_CFLAGS="-DWEBRTC_LINUX -DWEBRTC_THREAD_RR"
         PLATFORM_CFLAGS="-DWEBRTC_POSIX"
         OS_LDFLAGS="-lrt -lpthread"
         HAVE_POSIX=1
        ],
    [*-*darwin*],
        [
         OS_CFLAGS="-DWEBRTC_MAC -DWEBRTC_THREAD_RR -DWEBRTC_CLOCK_TYPE_REALTIME"
         AS_IF([test "$HAVE_IOS" = "yes"],
             [OS_CFLAGS+=" -DWEBRTC_IOS"])
         PLATFORM_CFLAGS="-DWEBRTC_POSIX"
         HAVE_POSIX=1
        ],
    [*-mingw32*],
        [
         OS_LDFLAGS="-lwinmm"
         PLATFORM_CFLAGS="-DWEBRTC_WIN -D_WIN32 -U__STRICT_ANSI__"
         HAVE_WIN=1
        ],
    [AC_MSG_ERROR([Unsupported host $host])]
)
AC_SUBST(PLATFORM_CFLAGS)
AM_CONDITIONAL(HAVE_POSIX, [test "x${HAVE_POSIX}" = "x1"])
AM_CONDITIONAL(HAVE_WIN, [test "x${HAVE_WIN}" = "x1"])

# Testing __ARM_ARCH_ISA_ARM since the code contains ARM instructions,
# which don't work on Thumb-2 only platforms (ARMv7-M).
AC_CHECK_DECLS([__ARM_ARCH_ISA_ARM],
	[HAVE_ARM=1; ARCH_CFLAGS="${ARCH_CFLAGS} -DWEBRTC_ARCH_ARM"])
AC_CHECK_DECLS([__ARM_ARCH_7A__],
	[HAVE_ARMV7=1; ARCH_CFLAGS="${ARCH_CFLAGS} -DWEBRTC_ARCH_ARM_V7"])
AC_CHECK_DECLS([__aarch64__],
	[HAVE_NEON=1; ARCH_CFLAGS="${ARCH_CFLAGS} -DWEBRTC_HAS_NEON -DWEBRTC_ARCH_ARM64"])
AC_CHECK_DECLS([__i386__], [HAVE_X86=1])
AC_CHECK_DECLS([__x86_64__], [HAVE_X86=1])

AM_CONDITIONAL(HAVE_X86, [test "x${HAVE_X86}" = "x1"])
AM_CONDITIONAL(HAVE_ARM, [test "x${HAVE_ARM}" = "x1"])
AM_CONDITIONAL(HAVE_ARMV7, [test "x${HAVE_ARMV7}" = "x1"])

# Borrowed from pulseaudio's configure.ac
AC_ARG_ENABLE([neon],
    AS_HELP_STRING([--enable-neon], [Enable NEON optimisations on ARM CPUs that support it (yes|no|auto|runtime)]))

AS_IF([test "x$enable_neon" != "xno"],
    AS_IF([test "x$enable_neon" != "xruntime"],
        [
         save_CXXFLAGS="$CXXFLAGS"; CXXFLAGS="-mfpu=neon $CXXFLAGS"
         AC_COMPILE_IFELSE(
            [AC_LANG_PROGRAM(
                [
                 #include <arm_neon.h>
                ], [])],
            [
             HAVE_NEON=1
             ARCH_CFLAGS="$ARCH_CFLAGS -DWEBRTC_HAS_NEON -mfpu=neon"
            ])
         CXXFLAGS="$save_CXXFLAGS"
        ],
        [
         HAVE_NEON=1
         ARCH_CFLAGS="$ARCH_CFLAGS -DWEBRTC_DETECT_NEON -mfpu=neon"
        ])
)
AM_CONDITIONAL([HAVE_NEON], [test "x$HAVE_NEON" = "x1"])

COMMON_CFLAGS="-DWEBRTC_AUDIO_PROCESSING_ONLY_BUILD ${PLATFORM_CFLAGS} ${OS_CFLAGS} ${ARCH_CFLAGS} -DNDEBUG -I\$(top_srcdir)"
COMMON_CXXFLAGS="-std=c++11 -DWEBRTC_AUDIO_PROCESSING_ONLY_BUILD ${PLATFORM_CFLAGS} ${OS_CFLAGS} ${ARCH_CFLAGS} ${GNUSTL_CFLAGS} -DNDEBUG -I\$(top_srcdir)"
COMMON_LDFLAGS="${OS_LDFLAGS}"
AC_SUBST([COMMON_CFLAGS])
AC_SUBST([COMMON_CXXFLAGS])
AC_SUBST([COMMON_LDFLAGS])

AC_CONFIG_FILES([
webrtc-audio-processing.pc
Makefile
webrtc/Makefile
webrtc/base/Makefile
webrtc/common_audio/Makefile
webrtc/system_wrappers/Makefile
webrtc/modules/Makefile
webrtc/modules/audio_coding/Makefile
webrtc/modules/audio_processing/Makefile
])

AC_OUTPUT
