from HierASR import HierASR
f = HierASR('HierASR/SR48k.pth')
f.run('audio.wav', 'aud_48.wav')