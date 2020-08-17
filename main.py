import convnet as cnn


net = cnn.ConvNet()
(lvh, lvs, lvv), (hvh, hvs, hvv) = net.set_camera()
net.predict_camera((lvh, lvs, lvv), (hvh, hvs, hvv))