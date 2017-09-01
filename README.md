
# WebRTC AEC AGC ANC NS示例
```
bash run.sh
```

For NS algorithm, you may refer to:
http://blog.csdn.net/shichaog/article/details/52514816

For VAD algorithm, you may refer to:
http://blog.csdn.net/shichaog/article/details/52399354

For AEC algorithm, you may refer to:
fullaec.m aec.jpg and the Paper, more over you can refer to my blog:
http://blog.csdn.net/shichaog/article/details/71152743

For beamforming:
	There is no demo in my example code, It's algorithm is simple.
It relies on delay-sum(Phase alignment for source direction), There can be no inteference to enable robustness(suppression direction).
It also relies on a diffusion nosie model.
All weight are represet in frequency domain, and there is large distortion in low and high frequecy, beacuse of using a Bandwith average value.

Welcome anybody who can participate in this project for other algorithm developing.

For more test audio source, you may connect me by shichaog@126.com.

I hope this project can help any body who want to know the principle of audio processing.

