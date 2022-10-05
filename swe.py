import math
import numpy as np
import matplotlib.pyplot as plt

# Parameters
t_final = 221
J = 100
dx = 1 / J
# dt = 0.005
dt = 0.008

# Vel and Height fields
velx = np.zeros((J + 1, J))
velx_post = np.zeros((J + 1, J))
vely = np.zeros((J, J + 1))
vely_post = np.zeros((J, J + 1))
height = np.zeros((J, J))
height_post = np.zeros((J, J))

#Initial wave
for x in range(40, 60):
    for y in range(40, 60):
        height[x, y] = 0.1

# plot setup
fig = plt.figure()
ax3 = plt.axes(projection='3d')

xx = np.arange(0, 1.00, 0.01)
yy = np.arange(0, 1.00, 0.01)
X, Y = np.meshgrid(xx, yy)

def interpolate_velx(x, y):
    assert x >= 0 and x <= J, 'In interpolate_velx, input x is %f' % (x)
    assert y >= 0 and y <= J, 'In interpolate_velx, input y is %f' % (y)

    x_base = int(math.floor(x))
    y_base = int(math.floor(y - 0.5))

    x_w = 1 - (x - x_base)
    y_w = 1 - ((y - 0.5) - y_base)

    ux = x_w * y_w * velx[x_base, y_base] + (1 - x_w) * y_w * velx[x_base + 1, y_base] + x_w * (1 - y_w) * velx[x_base, y_base + 1] + (1 - x_w) * (1 - y_w) * velx[x_base + 1, y_base + 1]
    return ux

def interpolate_vely(x, y):
    assert x >= 0 and x <= J, 'In interpolate_vely, input x is %f' % (x)
    assert y >= 0 and y <= J, 'In interpolate_vely, input y is %f' % (y)

    x_base = int(math.floor(x - 0.5))
    y_base = int(math.floor(y))

    x_w = 1 - ((x - 0.5) - x_base)
    y_w = 1 - (y - y_base)

    uy = x_w * y_w * vely[x_base, y_base] + (1 - x_w) * y_w * vely[x_base + 1, y_base] + x_w * (1 - y_w) * vely[x_base, y_base + 1] + (1 - x_w) * (1 - y_w) * vely[x_base + 1, y_base + 1]
    return uy

if __name__ == '__main__':
    for t in range(t_final):
        if t % 30 == 0:
            ax3.set_zlim(0, 1)
            ax3.plot_surface(X, Y, height, rstride = 1, cstride = 1, cmap = 'rainbow')
            plt.savefig('./swe_' + str(t) + '.png')
            plt.cla()

        # Backward advection of velx
        for y in range(1, J - 1):
            for x in range(1, J):
                ux = velx[x, y]
                uy = interpolate_vely(x, y + 0.5)

                prev_x = x - (ux * dt) / dx
                prev_y = (y + 0.5) - (uy * dt) / dx
                
                velx_post[x, y] = interpolate_velx(prev_x, prev_y)

        # Backward advection of vely
        for y in range(1, J):
            for x in range(1, J - 1):
                ux = interpolate_velx(x + 0.5, y)
                uy = vely[x, y]

                prev_x = (x + 0.5) - (ux * dt) / dx
                prev_y = y - (uy * dt) / dx

                vely_post[x, y] = interpolate_vely(prev_x, prev_y)

        # Height Integration
        for y in range(1, J - 1):
            for x in range(1, J - 1):
                # h(x + 0.5, y)
                h_bar1 = height[x + 1, y] if velx_post[x + 1, y] <= 0 else height[x, y]
                # h(x - 0.5, y)
                h_bar2 = height[x - 1, y] if velx_post[x, y] >= 0 else height[x, y]
                # h(x, y + 0.5)
                h_bar3 = height[x, y + 1] if vely_post[x, y + 1] <= 0 else height[x, y]
                # h(x, y - 0.5)
                h_bar4 = height[x, y - 1] if vely_post[x, y] >= 0 else height[x, y]

                dhdt = - (h_bar1 * velx_post[x + 1, y] - h_bar2 * velx_post[x, y]) / dx - (h_bar3 * vely_post[x, y + 1] - h_bar4 * vely_post[x, y]) / dx
                height_post[x, y] = height[x, y] + dhdt * dt

        # Velocity (x) Integration
        for y in range(1, J - 1):
            for x in range(1, J):
                velx_post[x, y] += (-10 / dx * (height_post[x, y] - height_post[x - 1, y])) * dt

        # Velocity (y) Integration
        for y in range(1, J):
            for x in range(1, J - 1):
                vely_post[x, y] += (-10 / dx * (height_post[x, y] - height_post[x, y - 1])) * dt

        # Exchange the pointers
        velx_temp = velx
        velx = velx_post
        velx_post= velx_temp

        vely_temp = vely
        vely = vely_post
        vely_post= vely_temp

        height_temp = height
        height = height_post
        height_post= height_temp