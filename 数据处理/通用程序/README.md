# 光纤远场分布
拿来分析CP光纤出来光的
1. [光纤远场分布](https://zhuanlan.zhihu.com/p/624724096)
$I(r, z) = \frac{2P}{\pi \omega^2(z)} \exp \left[ \frac{-2r^2}{\omega^2(z)} \right]$
$z_R = \frac{\pi \omega_0^2}{\lambda}$
$\omega(z) = \omega_0 \sqrt{1 + \left( \frac{z}{z_R} \right)^2}$
$\theta = \frac{\lambda}{\pi \omega_0}$
然后又有几何关系
$r = [(x' \cos \alpha)^2 + (y')^2]^{1/2}$
$z_0 - z = x' \sin \alpha$
$x' = \cos \beta (x-x_0) + \sin \beta (y-y_0)$
$y' = -\sin \beta (x-x_0) + \cos \beta (y-y_0)$

2. 所有像素根据光被针遮挡和无遮挡分开，再根据光强分析中心点，最后再尝试用远场分布分析光强