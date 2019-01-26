# Now compute the standard deviation by calculating the
# square root of the expected squared differences
std_img_op = tf.sqrt(tf.reduce_mean(subtraction * subtraction, axis=0))

# Now calculate the standard deviation using your session
std_img = sess.run(std_img_op)

norm_imgs_op = tf.convert_to_tensor(imgs) - mean_img

norm_imgs = sess.run(norm_imgs_op)
print(np.min(norm_imgs), np.max(norm_imgs))
print(imgs.dtype)

# Then plot the resulting normalized dataset montage:
# Make sure we have a 100 x 100 x 100 x 3 dimension array
assert(norm_imgs.shape == (100, 100, 100, 3))
plt.figure(figsize=(10, 10))
plt.imshow(utils.montage(norm_imgs, 'normalized.png'))

norm_imgs_show = (norm_imgs - np.min(norm_imgs)) / (np.max(norm_imgs) - np.min(norm_imgs))
plt.figure(figsize=(10, 10))
plt.imshow(utils.montage(norm_imgs_show, 'normalized.png'))

# First build 3 kernels for each input color channel
ksize = 3
kernel = np.concatenate([utils.gabor(ksize)[:, :, np.newaxis] for i in range(3)], axis=2)

# Now make the kernels into the shape: [ksize, ksize, 3, 1]:
kernel_4d = tf.reshape(kernel, [3, 3, 3, 1])
assert(kernel_4d.shape == (ksize, ksize, 3, 1))

sess = tf.Session()
plt.figure(figsize=(5, 5))
plt.imshow(sess.run(kernel_4d[:, :, 0, 0]), cmap='gray')
plt.imsave(arr=kernel_4d[:, :, 0, 0], fname='kernel.png', cmap='gray')
