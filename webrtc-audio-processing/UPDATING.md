Updating
=====

Assembling some quick notes on maintaining this tree vs. the upstream WebRTC
project source code.

1. The code is currently synced agains whatever revision of the upstream
   webrtc git repository Chromium uses.

2. Instructions on checking out the Chromium tree are on the
   [Chromium site][get-chromium]. As a shortcut, you can look at the DEPS file
   in the Chromium tree for the current webrtc version being used, and then
   just use that commit hash with the webrtc tree.

3. [Meld][meld] is a great tool for diffing two directories. Start by running
   it on ```webrtc-audio-processing/webrtc``` and
   ```chromium/third_party/webrtc```.

   * For each directory in the ```webrtc-audio-processing``` tree, go over the
     corresponding code in the ```chromium``` tree.

   * Examine changed files, and pick any new changes. A small number of files
     in the ```webrtc-audio-processing``` tree have been changed by hand, make
     sure that those are not overwritten.

   * unittest files have been left out since they are not built or used.

   * BUILD.gn files have been copied to keep track of changes to the build
     system upstreama.

   * Arch-specific files usually have special handling in the corresponding
     Makefile.am.

4. Once everything has been copied and updated, everything needs to be built.
   Missing dependencies (files that were not copied, or new modules that are
   being depended on) will first turn up here.

   * Copy new deps as needed, leaving out testing-only dependencies insofar as
     this is possible.

5. ```webrtc/modules/audio_processing/include/audio_processing.h``` is the main
   include file, so look for API changes here.

   * The current policy is that we mirror upstream API as-is.

   * Update configure.ac with the appropriate version info  based on how the
     code has changed. Details on how to do this are included in the
     [libtool documentation][libtool-version-info].

5. Build PulseAudio (and/or any other dependent projects) against the new code.
   The easy way to do this is via a prefixed install.

   * Run ```configure``` webrtc-audio-processing with
     ```--prefix=/some/local/path```, then do a ```make``` and
     ```make install```.

   * Run ```configure``` on PulseAudio with
     ```PKG_CONFIG_PATH=/some/local/path/lib/pkgconfig```, which will cause the
     build to pick up the prefixed install. Then do a ```make```, run the built
     PulseAudio, and load ```module-echo-cancel``` to make sure it loads fine.

   * Run some test streams through the canceller to make sure it is working
     fine.

[get-chromium]: http://dev.chromium.org/developers/how-tos/get-the-code
[meld]: http://meldmerge.org/
[libtool-version-info]: https://www.gnu.org/software/libtool/manual/html_node/Updating-version-info.html
