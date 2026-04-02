import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.animation import FuncAnimation

# --- Core Settings ---
L1 = 0.4
L2 = 0.4
NUM_DISKS_1 = 8
NUM_DISKS_2 = 8
DISK_RADIUS = 0.025

def calc_transform(k, phi, s):
    """Calculates 3D kinematics for the tail."""
    T = np.eye(4)
    if abs(k) < 1e-4:
        T[2, 3] = s
        return T
    
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_ks, s_ks = np.cos(k*s), np.sin(k*s)
    
    T[0,0] = c_phi**2 * c_ks + s_phi**2
    T[0,1] = s_phi*c_phi*(c_ks - 1)
    T[0,2] = c_phi*s_ks
    T[0,3] = c_phi*(1 - c_ks)/k
    
    T[1,0] = s_phi*c_phi*(c_ks - 1)
    T[1,1] = c_phi**2 + s_phi**2 * c_ks
    T[1,2] = s_phi*s_ks
    T[1,3] = s_phi*(1 - c_ks)/k
    
    T[2,0] = -c_phi*s_ks
    T[2,1] = -s_phi*s_ks
    T[2,2] = c_ks
    T[2,3] = s_ks/k
    return T

def draw_disk(ax, transform, color):
    theta = np.linspace(0, 2*np.pi, 15)
    circle = np.array([DISK_RADIUS * np.cos(theta), DISK_RADIUS * np.sin(theta), np.zeros_like(theta), np.ones_like(theta)])
    transformed_circle = transform @ circle
    ax.plot(transformed_circle[0,:], transformed_circle[1,:], transformed_circle[2,:], color=color, linewidth=1.5)

# --- Setup Plot and GUI ---
fig = plt.figure(figsize=(12, 8))
plt.subplots_adjust(left=0.3, bottom=0.35) # Make room for UI on left and bottom
ax = fig.add_subplot(111, projection='3d')

# --- UI Controls ---
axcolor = 'lightgoldenrodyellow'
ax_bx = plt.axes([0.3, 0.20, 0.60, 0.03], facecolor=axcolor)
ax_by = plt.axes([0.3, 0.15, 0.60, 0.03], facecolor=axcolor)
ax_tx = plt.axes([0.3, 0.10, 0.60, 0.03], facecolor=axcolor)
ax_ty = plt.axes([0.3, 0.05, 0.60, 0.03], facecolor=axcolor)

slider_base_x = Slider(ax_bx, 'Base Bend X', -8.0, 8.0, valinit=0.0)
slider_base_y = Slider(ax_by, 'Base Bend Y', -8.0, 8.0, valinit=0.0)
slider_tip_x = Slider(ax_tx, 'Tip Bend X', -8.0, 8.0, valinit=0.0)
slider_tip_y = Slider(ax_ty, 'Tip Bend Y', -8.0, 8.0, valinit=0.0)

# Radio Buttons for Modes
ax_radio = plt.axes([0.05, 0.4, 0.15, 0.25], facecolor=axcolor)
radio = RadioButtons(ax_radio, ('Manual', 'Cat Tail', 'Grasping', 'Helicopter'))
current_mode = 'Manual'

def mode_switch(label):
    global current_mode
    current_mode = label
radio.on_clicked(mode_switch)

def draw_tail(bx, by, tx, ty):
    """Core drawing routine"""
    ax.clear()
    ax.set_xlim([-0.6, 0.6])
    ax.set_ylim([-0.6, 0.6])
    ax.set_zlim([0, 0.9])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height (m)')
    ax.set_title(f'Robotic Tail - Mode: {current_mode}')

    k1, phi1 = np.sqrt(bx**2 + by**2), np.arctan2(by, bx)
    k2, phi2 = np.sqrt(tx**2 + ty**2), np.arctan2(ty, tx)

    spine_x, spine_y, spine_z = [], [], []

    # Section A
    for s in np.linspace(0, L1, NUM_DISKS_1):
        T1 = calc_transform(k1, phi1, s)
        spine_x.append(T1[0,3]); spine_y.append(T1[1,3]); spine_z.append(T1[2,3])
        draw_disk(ax, T1, 'blue')
    
    T_end_1 = calc_transform(k1, phi1, L1)

    # Section B
    for s in np.linspace(L1/NUM_DISKS_2, L2, NUM_DISKS_2):
        T_combined = T_end_1 @ calc_transform(k2, phi2, s)
        spine_x.append(T_combined[0,3]); spine_y.append(T_combined[1,3]); spine_z.append(T_combined[2,3])
        draw_disk(ax, T_combined, 'orange')

    ax.plot(spine_x, spine_y, spine_z, color='black', linewidth=3)

def manual_update(val=None):
    if current_mode == 'Manual':
        draw_tail(slider_base_x.val, slider_base_y.val, slider_tip_x.val, slider_tip_y.val)
        fig.canvas.draw_idle()

slider_base_x.on_changed(manual_update)
slider_base_y.on_changed(manual_update)
slider_tip_x.on_changed(manual_update)
slider_tip_y.on_changed(manual_update)

# --- Animation Engine ---
time_var = 0.0
def animate(frame):
    global time_var
    if current_mode == 'Manual':
        return # Do nothing, let user control sliders
    
    time_var += 0.1 # Game tick
    t = time_var
    bx, by, tx, ty = 0.0, 0.0, 0.0, 0.0
    
    if current_mode == 'Cat Tail':
        # Organic wave: tip follows the base with a delay
        bx = 2.5 * np.sin(t)
        tx = 3.5 * np.sin(t - 1.0) # Phase shift creates the "whip" look
    
    elif current_mode == 'Grasping':
        # Grab and release: Uses max() to create a resting period
        curl = max(0.0, np.sin(t * 1.5)) * 6.0 
        bx = curl           # Base bends forward
        tx = -curl * 1.2    # Tip hooks backward harder to wrap around
        
    elif current_mode == 'Helicopter':
        # Circular motion using sine and cosine on both axes
        bx = 3.0 * np.cos(t)
        by = 3.0 * np.sin(t)
        tx = 5.0 * np.cos(t)
        ty = 5.0 * np.sin(t)

    # Visually update the sliders so you can see the math happening!
    # (We turn off events temporarily so they don't fight the animation loop)
    for slider, val in zip([slider_base_x, slider_base_y, slider_tip_x, slider_tip_y], [bx, by, tx, ty]):
        slider.eventson = False
        slider.set_val(val)
        slider.eventson = True

    draw_tail(bx, by, tx, ty)

# Run animation at roughly 20 FPS (50ms interval)
ani = FuncAnimation(fig, animate, interval=50, blit=False)

# Initial draw
draw_tail(0, 0, 0, 0)
plt.show()