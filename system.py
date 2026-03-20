import pygame
import numpy as np
from simple_pid import PID
import math
import random
import time

# Ensure you have the following libraries installed:
# pip install pygame numpy simple-pid

class CubeSat:
    """
    Represents the CubeSat model and its physical state, including orbital dynamics.
    """
    def __init__(self, mass=1.33, inertia_matrix=None, initial_position=None, initial_velocity=None, initial_attitude_rad=None, initial_angular_velocity=None):
        self.mass = mass  # Mass in kg
        self.inertia_matrix = inertia_matrix if inertia_matrix is not None else np.diag([0.01, 0.01, 0.01])
        
        # Orbital state vectors (in meters and m/s)
        self.position = np.array(initial_position, dtype=float) if initial_position is not None else np.array([7000000.0, 0.0, 0.0])
        self.velocity = np.array(initial_velocity, dtype=float) if initial_velocity is not None else np.array([0.0, 7000.0, 0.0])

        # Attitude state vectors (using Euler angles in radians)
        self.attitude = np.array(initial_attitude_rad, dtype=float) if initial_attitude_rad is not None else np.zeros(3)
        self.angular_velocity = np.array(initial_angular_velocity, dtype=float) if initial_angular_velocity is not None else np.zeros(3)

    def update_attitude(self, applied_torque, dt):
        """Updates the CubeSat's attitude based on an applied torque and a time step."""
        angular_acceleration = np.linalg.inv(self.inertia_matrix) @ applied_torque
        self.angular_velocity += angular_acceleration * dt
        self.attitude += self.angular_velocity * dt
        
        # Keep attitude angles within the range of -pi to pi
        self.attitude = np.fmod(self.attitude + np.pi, 2 * np.pi) - np.pi

    def update_orbit(self, total_force, dt):
        """Updates the CubeSat's position and velocity based on total applied force."""
        acceleration = total_force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

class AttitudeController:
    """
    Implements a simple PID controller for attitude stabilization.
    """
    def __init__(self, kp, ki, kd, setpoint):
        # Create PID controllers for each axis (roll, pitch, yaw)
        self.pid_roll = PID(kp, ki, kd, setpoint=setpoint[0])
        self.pid_pitch = PID(kp, ki, kd, setpoint=setpoint[1])
        self.pid_yaw = PID(kp, ki, kd, setpoint=setpoint[2])

        # Set the output limits for torque
        self.pid_roll.output_limits = (-0.01, 0.01)
        self.pid_pitch.output_limits = (-0.01, 0.01)
        self.pid_yaw.output_limits = (-0.01, 0.01)
        
        # Set Sample time to None to use external time step in the simulation loop
        self.pid_roll.sample_time = None
        self.pid_pitch.sample_time = None
        self.pid_yaw.sample_time = None

    def calculate_torque(self, current_attitude):
        """Calculates the required torque to reach the setpoint."""
        torque_x = self.pid_roll(current_attitude[0])
        torque_y = self.pid_pitch(current_attitude[1])
        torque_z = self.pid_yaw(current_attitude[2])
        return np.array([torque_x, torque_y, torque_z])

class OrbitalController:
    """
    Implements a PID controller to maintain a stable circular orbit.
    """
    def __init__(self, kp, ki, kd, setpoint_radius):
        self.pid_radius = PID(kp, ki, kd, setpoint=setpoint_radius)
        self.pid_radius.output_limits = (-0.1, 0.1) # Max thrust in Newtons
        self.pid_radius.sample_time = None # Use external time step

    def calculate_thrust(self, current_position):
        """Calculates the required thrust to correct the orbit."""
        current_radius = np.linalg.norm(current_position)
        error = current_radius - self.pid_radius.setpoint
        
        thrust_magnitude = self.pid_radius(current_radius)

        if np.abs(error) > 1000: # Only apply thrust if there is a significant error (1 km)
            thrust_direction = -current_position / current_radius # Towards the center of the orbit
            return thrust_magnitude * thrust_direction
        
        return np.zeros(3)

class CubeSatSimulator:
    """
    Main application class for the CubeSat simulator GUI using Pygame.
    """
    def __init__(self):
        pygame.init()
        self.screen_width = 1200
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
        pygame.display.set_caption("CubeSat Attitude and Orbital Control Simulator")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 18)
        self.speed_font = pygame.font.SysFont('Arial', 24)
        self.running = True

        self.colors = {
            'background': (0, 0, 0), 'star': (255, 255, 255), 'text': (255, 255, 255),
            'panel_bg': (20, 20, 20), 'panel_border': (150, 150, 150),
            'button_text': (255, 255, 255), 'button_bg': (50, 50, 50)
        }
        
        self.PLANET_MODELS = [
            {'name': 'Earth', 'mass': 5.972e24, 'radius': 6371000.0, 'color': (0, 0, 150), 'land_color': (0, 50, 0)},
            {'name': 'Mars', 'mass': 6.39e23, 'radius': 3389500.0, 'color': (150, 50, 0), 'land_color': (200, 100, 50)},
            {'name': 'Jupiter', 'mass': 1.898e27, 'radius': 69911000.0, 'color': (100, 70, 0), 'land_color': (150, 120, 50)},
            {'name': 'Venus', 'mass': 4.867e24, 'radius': 6051800.0, 'color': (255, 140, 0), 'land_color': (255, 165, 0)},
            {'name': 'Saturn', 'mass': 5.683e26, 'radius': 58232000.0, 'color': (244, 230, 130), 'land_color': (210, 180, 140)}
        ]
        self.current_planet_index = 0

        self.cube_sat = None
        self.attitude_controller = None
        self.orbital_controller = None
        
        self.state = 'LOGIN'
        self.simulation_state = 'STOPPED'

        self.simulation_time = 0.0
        self.simulation_speed_factor = 1.0
        self.sim_history, self.off_attitude_events = [], []
        self.history_scroll_offset, self.off_attitude_scroll_offset = 0, 0
        self.warning_message = ""
        self.show_history, self.is_satellite_visible, self.show_off_attitude_history = True, True, True
        self.max_history_size = 1000
        
        self.G = 6.67430e-11
        self.orbital_scale = 1.0
        self.view_center = np.array([self.screen_width / 2.0, self.screen_height / 2.0])
        
        self.cube_vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ]) * 10
        self.cube_faces = [
            (0, 1, 2, 3), (4, 5, 6, 7), (0, 4, 7, 3),
            (1, 5, 6, 2), (0, 1, 5, 4), (3, 2, 6, 7)
        ]
        self.stars = self.generate_stars()
        
        self.rects = {} # Dictionary to hold all UI rects

        self.show_planet_menu, self.show_planet_details = False, False
        self.selected_planet_for_details = None

        self.username_input, self.password_input = '', ''
        self.active_input = 'username'
        self.login_error_message = ''
        self.show_loading_bar = False
        self.loading_progress, self.loading_start_time = 0, 0
        self.loading_duration = 3.0
        
        self.login_satellite_angle = 0.0
        
        self.cursor_visible = True
        self.cursor_blink_interval = 500
        self.last_cursor_toggle = pygame.time.get_ticks()
        self.show_username, self.show_password = False, False
        self.stability_status = "Initializing"
        self.is_stabilizing = False
        self.manual_disturbance_applied = False

        self.initial_attitude = np.radians(np.random.uniform(-5, 5, 3)) 
        self.update_simulation_params()
        self.update_ui_elements()

    def update_simulation_params(self):
        """Updates planet-dependent parameters and resets the simulation."""
        current_planet = self.PLANET_MODELS[self.current_planet_index]
        self.planet_mass = current_planet['mass']
        self.planet_radius = current_planet['radius']
        
        target_pixel_radius = min(self.screen_width, self.screen_height) * 0.25
        self.orbital_scale = target_pixel_radius / self.planet_radius
        self.planet_pixel_radius = int(self.planet_radius * self.orbital_scale)
        
        self.target_orbit_radius = self.planet_radius + 600000.0

        self.initial_pos = np.array([self.target_orbit_radius, 0.0, 0.0])
        self.initial_vel = np.array([0.0, math.sqrt(self.G * self.planet_mass / self.target_orbit_radius), 0.0])

        self.reset_simulation()

    def generate_stars(self):
        """Generates a list of random points for the starry background."""
        return [(random.randint(0, self.screen_width), random.randint(0, self.screen_height)) for _ in range(500)]

    def update_ui_elements(self):
        """Recalculates positions of all UI elements based on current screen size."""
        padding, button_width, button_height, panel_width = 20, 150, 50, 300
        
        panel_x = self.screen_width - panel_width - padding
        self.rects['graph'] = pygame.Rect(panel_x, padding, panel_width, 200)
        self.rects['off_attitude'] = pygame.Rect(panel_x, self.rects['graph'].bottom + padding, panel_width, 200)
        
        telemetry_height = self.screen_height - self.rects['off_attitude'].bottom - (3 * padding) - button_height
        self.rects['telemetry'] = pygame.Rect(panel_x, self.rects['off_attitude'].bottom + padding, panel_width, telemetry_height)

        control_y = self.screen_height - button_height - padding
        
        self.rects['start_resume_button'] = pygame.Rect(padding, control_y, button_width, button_height)
        self.rects['pause_button'] = pygame.Rect(padding + button_width + 10, control_y, button_width, button_height)
        self.rects['stop_button'] = pygame.Rect(padding + (button_width + 10) * 2, control_y, button_width, button_height)
        self.rects['reset_button'] = pygame.Rect(padding + (button_width + 10) * 3, control_y, button_width, button_height)

        self.rects['change_planet_button'] = pygame.Rect(self.screen_width - padding - button_width, control_y, button_width, button_height)
        self.rects['toggle_off_attitude_button'] = pygame.Rect(self.rects['change_planet_button'].left - 10 - button_width, control_y, button_width, button_height)
        self.rects['toggle_satellite_button'] = pygame.Rect(self.rects['toggle_off_attitude_button'].left - 10 - button_width, control_y, button_width, button_height)
        self.rects['toggle_history_button'] = pygame.Rect(self.rects['toggle_satellite_button'].left - 10 - button_width, control_y, button_width, button_height)
        
        self.rects['slow_down_button'] = pygame.Rect(padding, padding + 60, 50, 50)
        self.rects['speed_up_button'] = pygame.Rect(padding + 60, padding + 60, 50, 50)
        
        self.rects['stabilize_button'] = pygame.Rect(self.screen_width/2 - 75, 70, button_width, button_height-10)

        menu_width, menu_height = 250, len(self.PLANET_MODELS) * 30 + 60
        self.rects['planet_menu'] = pygame.Rect((self.screen_width - menu_width) / 2, (self.screen_height - menu_height) / 2, menu_width, menu_height)

        details_width, details_height = 350, 220
        self.rects['planet_details'] = pygame.Rect((self.screen_width - details_width) / 2, (self.screen_height - details_height) / 2, details_width, details_height)
        self.rects['close_details_button'] = pygame.Rect(self.rects['planet_details'].right - 40, self.rects['planet_details'].top + 10, 30, 30)
        self.rects['select_planet_button'] = pygame.Rect(self.rects['planet_details'].centerx - 75, self.rects['planet_details'].bottom - 50, 150, 40)
        
        self.rects['username_toggle'] = pygame.Rect(self.screen_width/2 + 155, self.screen_height/2 - 60, 30, 30)
        self.rects['password_toggle'] = pygame.Rect(self.screen_width/2 + 155, self.screen_height/2 + 40, 30, 30)
        self.rects['username_field'] = pygame.Rect(self.screen_width/2 - 50, self.screen_height/2 - 60, 200, 30)
        self.rects['password_field'] = pygame.Rect(self.screen_width/2 - 50, self.screen_height/2 + 40, 200, 30)
        self.rects['login_button'] = pygame.Rect(self.screen_width/2 - 75, self.screen_height/2 + 120, 150, 40)
        self.view_center = np.array([self.screen_width / 2.0, self.screen_height / 2.0])

    def start_simulation(self):
        """Initializes and starts the simulation."""
        if self.simulation_state == 'RUNNING': return
        
        self.cube_sat = CubeSat(initial_position=self.initial_pos, initial_velocity=self.initial_vel, initial_attitude_rad=self.initial_attitude)
        self.attitude_controller = AttitudeController(kp=0.01, ki=0.0001, kd=0.02, setpoint=np.radians([0, 0, 0]))
        self.orbital_controller = OrbitalController(kp=0.0001, ki=0.00001, kd=0.0002, setpoint_radius=self.target_orbit_radius)
        self.simulation_state = 'RUNNING'
        self.sim_history, self.off_attitude_events = [], []
        self.simulation_time, self.simulation_speed_factor = 0.0, 1.0
        self.warning_message = ""

    def reset_simulation(self):
        """Resets the simulation to its initial state."""
        self.simulation_state = 'STOPPED'
        self.cube_sat = None
        self.simulation_time = 0.0
        self.sim_history, self.off_attitude_events = [], []
        self.history_scroll_offset, self.off_attitude_scroll_offset = 0, 0
        self.warning_message, self.simulation_speed_factor = "", 1.0
        self.show_planet_details, self.show_planet_menu = False, False
        self.initial_attitude = np.radians(np.random.uniform(-5, 5, 3))
        self.stability_status = "Initializing"
        self.is_stabilizing, self.manual_disturbance_applied = False, False
        self.update_ui_elements()

    def calculate_gravity(self, position):
        """Calculates the gravitational force vector."""
        r = np.linalg.norm(position)
        if r == 0: return np.zeros(3)
        force_magnitude = (self.G * self.planet_mass * self.cube_sat.mass) / (r**2)
        return -position / r * force_magnitude
    
    def draw_text(self, text, pos, color=None, font=None):
        """Helper function to render text on the screen."""
        color = color or self.colors['text']
        font = font or self.font
        return font.render(text, True, color)

    def draw_planet(self):
        """Draws the central planet with a simple 2D effect."""
        current_planet = self.PLANET_MODELS[self.current_planet_index]
        color, land_color, name = current_planet['color'], current_planet['land_color'], current_planet['name']
        
        pygame.draw.circle(self.screen, color, self.view_center, self.planet_pixel_radius)
        pygame.draw.circle(self.screen, land_color, self.view_center, int(self.planet_pixel_radius * 0.85))
        
        self.screen.blit(self.draw_text(name, (0, 0)), self.view_center - np.array([len(name)*5, self.planet_pixel_radius + 20]))
        
        target_orbit_pixel_radius = int(self.target_orbit_radius * self.orbital_scale)
        pygame.draw.circle(self.screen, (0, 255, 0), self.view_center, target_orbit_pixel_radius, 1)

    def draw_cube(self, position, attitude, scale_factor=1.0):
        """Draws a 3D projected cube with rotation."""
        cube_screen_pos = self.view_center + (position[:2] * self.orbital_scale if self.state == 'SIMULATION' else np.array([0,0]))
        
        roll, pitch, yaw = attitude
        Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        rotation_matrix = Rz @ Ry @ Rx
        
        distance_scale = 1 + self.cube_sat.position[2] / (self.target_orbit_radius * 2) if self.state == 'SIMULATION' and self.cube_sat else 1.0
        final_scale = scale_factor * distance_scale
        
        projected_points = [(vertex @ rotation_matrix.T)[:2] * final_scale + cube_screen_pos for vertex in self.cube_vertices]
            
        face_colors = [(200,200,200), (150,150,150), (100,100,100), (50,50,50), (120,120,120), (80,80,80)]
        face_depths = [(np.mean([(self.cube_vertices[j] @ rotation_matrix.T)[2] for j in face]), i) for i, face in enumerate(self.cube_faces)]
            
        for _, i in sorted(face_depths, key=lambda x: x[0]):
            points = [tuple(projected_points[j]) for j in self.cube_faces[i]]
            pygame.draw.polygon(self.screen, face_colors[i], points)
            pygame.draw.polygon(self.screen, (0, 0, 0), points, 1)

    def draw_scrollable_panel(self, rect_key, title, title_color, data_list, line_generator):
        """Generic function to draw a scrollable panel."""
        rect = self.rects[rect_key]
        pygame.draw.rect(self.screen, self.colors['panel_bg'], rect)
        pygame.draw.rect(self.screen, title_color, rect, 2)
        self.screen.blit(self.draw_text(title, (0, 0), color=title_color), (rect.x + 10, rect.y + 10))

        if not data_list: return
        
        entry_height = line_generator(data_list[0], 0, None, measure_only=True)
        total_content_height = max(1, len(data_list)) * entry_height
        
        content_surface = pygame.Surface((rect.width - 20, total_content_height))
        content_surface.set_colorkey(self.colors['panel_bg'])
        content_surface.fill(self.colors['panel_bg'])

        for i, entry in enumerate(reversed(data_list)):
            y_pos = i * entry_height
            line_generator(entry, y_pos, content_surface)
        
        scroll_offset = self.history_scroll_offset if rect_key == 'telemetry' else self.off_attitude_scroll_offset
        view_rect = pygame.Rect(0, scroll_offset, rect.width - 20, rect.height - 40)
        self.screen.blit(content_surface, (rect.x + 10, rect.y + 40), view_rect)

        viewable_height = rect.height - 40
        if total_content_height > viewable_height:
            scroll_bar_height = viewable_height * (viewable_height / total_content_height)
            scroll_bar_y_ratio = scroll_offset / (total_content_height - viewable_height) if (total_content_height - viewable_height) > 0 else 0
            scroll_bar_y = rect.y + 40 + scroll_bar_y_ratio * (viewable_height - scroll_bar_height)
            scroll_bar_rect = pygame.Rect(rect.right - 10, scroll_bar_y, 8, scroll_bar_height)
            pygame.draw.rect(self.screen, (200, 200, 200), scroll_bar_rect)

    def telemetry_line_generator(self, entry, y_pos, surface, measure_only=False):
        line_height = 18
        if measure_only: return line_height * 3
        
        attitude_deg = np.degrees(entry['attitude']) 
        time_text = f"T: {entry['time']:.1f} s"
        radius = np.linalg.norm(entry['position'])
        pos_text = f"R: {radius/1000:.1f} km, Vel: {np.linalg.norm(entry['velocity']):.1f} m/s"
        att_text = f"Att: R {attitude_deg[0]:.1f} P {attitude_deg[1]:.1f} Y {attitude_deg[2]:.1f} deg"
        
        surface.blit(self.draw_text(time_text, (0, 0), (150, 150, 150)), (0, y_pos))
        surface.blit(self.draw_text(pos_text, (0, 0), color=(0, 255, 0)), (0, y_pos + line_height))
        surface.blit(self.draw_text(att_text, (0, 0), color=(0, 255, 0)), (0, y_pos + 2 * line_height))

    def off_attitude_line_generator(self, entry, y_pos, surface, measure_only=False):
        line_height = 25
        if measure_only: return line_height
        
        attitude_deg = np.degrees(entry['attitude'])
        time_text = f"T: {entry['time']:.1f}s"
        att_text = f"R {attitude_deg[0]:.1f}, P {attitude_deg[1]:.1f}, Y {attitude_deg[2]:.1f}"
        
        surface.blit(self.draw_text(time_text, (0, 0), (255, 255, 50)), (0, y_pos))
        surface.blit(self.draw_text(att_text, (0, 0), color=(255, 50, 50)), (70, y_pos))

    def draw_attitude_graphs(self):
        """Draws live attitude graphs for Roll, Pitch, and Yaw."""
        rect = self.rects['graph']
        pygame.draw.rect(self.screen, self.colors['panel_bg'], rect)
        pygame.draw.rect(self.screen, (0, 255, 255), rect, 2)
        self.screen.blit(self.draw_text("Live Attitude (deg)", (0, 0), color=(0, 255, 0)), (rect.x + 10, rect.y + 10))
        
        if len(self.sim_history) > 1:
            att_data_deg = np.degrees(np.array([e['attitude'] for e in self.sim_history[-100:]]))
            
            max_val = np.max(np.abs(att_data_deg)) if att_data_deg.size > 0 else 5.0
            if max_val < 5: max_val = 5
            
            h, w = rect.height - 60, rect.width - 10
            x_scale = w / max(1, len(att_data_deg) - 1)
            y_scale = h / (2 * max_val)
            y_offset = rect.y + 40 + h / 2
            
            pygame.draw.line(self.screen, (50, 50, 50), (rect.x, y_offset), (rect.right, y_offset), 1)
            
            points = {
                'Roll': ([(rect.x + 5 + i * x_scale, y_offset - att_data_deg[i, 0] * y_scale) for i in range(len(att_data_deg))], (255,0,0)),
                'Pitch': ([(rect.x + 5 + i * x_scale, y_offset - att_data_deg[i, 1] * y_scale) for i in range(len(att_data_deg))], (0,255,0)),
                'Yaw': ([(rect.x + 5 + i * x_scale, y_offset - att_data_deg[i, 2] * y_scale) for i in range(len(att_data_deg))], (0,0,255))
            }
            
            for i, (name, (pts, color)) in enumerate(points.items()):
                if len(pts) > 1: pygame.draw.lines(self.screen, color, False, pts, 2)
                self.screen.blit(self.draw_text(name, (0, 0), color=color), (rect.x + 10 + i * 50, rect.y + 30))
            
    def draw_speed_icons(self, rect_key, direction):
        """Draws a custom speed icon on the given rectangle."""
        rect = self.rects[rect_key]
        color, center_x, center_y = self.colors['text'], rect.centerx, rect.centery
        
        if direction == "rewind":
            points = [[(center_x, center_y), (center_x + 10, center_y - 10), (center_x + 10, center_y + 10)],
                      [(center_x - 10, center_y), (center_x, center_y - 10), (center_x, center_y + 10)]]
        else: # fast_forward
            points = [[(center_x, center_y), (center_x - 10, center_y - 10), (center_x - 10, center_y + 10)],
                      [(center_x + 10, center_y), (center_x, center_y - 10), (center_x, center_y + 10)]]
        for p in points: pygame.draw.polygon(self.screen, color, p)
    
    def draw_planet_menu(self):
        """Draws the planet selection menu."""
        rect = self.rects['planet_menu']
        pygame.draw.rect(self.screen, self.colors['panel_bg'], rect, border_radius=15)
        pygame.draw.rect(self.screen, self.colors['panel_border'], rect, 2, border_radius=15)
        
        title_text = self.draw_text("Select a Planet", (0,0), font=self.speed_font)
        self.screen.blit(title_text, title_text.get_rect(center=(rect.centerx, rect.top + 20)))

        for i, planet in enumerate(self.PLANET_MODELS):
            planet_rect = pygame.Rect(rect.x + 20, rect.y + 50 + i * 30, rect.width - 40, 25)
            text_color = (0, 255, 0) if i == self.current_planet_index else (255, 255, 255)
            self.screen.blit(self.draw_text(planet['name'], (0,0), color=text_color), planet_rect)
            self.rects[f"planet_menu_item_{i}"] = planet_rect

    def draw_planet_details_box(self):
        """Draws a pop-up box with details about the selected planet."""
        rect = self.rects['planet_details']
        pygame.draw.rect(self.screen, self.colors['panel_bg'], rect, border_radius=15)
        pygame.draw.rect(self.screen, self.colors['panel_border'], rect, 2, border_radius=15)

        planet = self.selected_planet_for_details
        if not planet: return

        try:
            surface_gravity = (self.G * planet['mass']) / (planet['radius']**2)
            gravity_text = f"Surface Gravity: {surface_gravity:.2f} m/s²"
        except ZeroDivisionError:
            gravity_text = "Surface Gravity: N/A"

        texts = [
            (planet['name'], self.speed_font, (rect.x + 20, rect.y + 20)),
            (f"Mass: {planet['mass']:.2e} kg", self.font, (rect.x + 20, rect.y + 60)),
            (f"Radius: {planet['radius']/1000:.0f} km", self.font, (rect.x + 20, rect.y + 90)),
            (gravity_text, self.font, (rect.x + 20, rect.y + 120))
        ]
        for text, font, pos in texts: self.screen.blit(self.draw_text(text, (0,0), font=font), pos)

        select_rect = self.rects['select_planet_button']
        pygame.draw.rect(self.screen, (0, 100, 0), select_rect)
        self.screen.blit(self.draw_text("Select This Planet", (0,0)), (select_rect.x + 10, select_rect.y + 10))

        close_rect = self.rects['close_details_button']
        pygame.draw.rect(self.screen, (100, 0, 0), close_rect)
        self.screen.blit(self.draw_text("X", (0,0)), (close_rect.x + 9, close_rect.y + 4))
        
    def draw_eye_icon(self, rect_key, is_hidden):
        """Draws an eye icon in the given rect."""
        rect = self.rects[rect_key]
        center, color = rect.center, (200, 200, 200)
        pygame.draw.ellipse(self.screen, color, rect.inflate(-10, -20), 1)
        pygame.draw.circle(self.screen, color, center, 4)
        if is_hidden: pygame.draw.line(self.screen, (255, 0, 0), rect.topleft, rect.bottomright, 2)

    def draw_login_screen(self):
        """Draws the login screen UI."""
        self.screen.fill(self.colors['background'])
        for star in self.stars: pygame.draw.circle(self.screen, self.colors['star'], star, random.randint(1, 2))

        self.draw_cube(np.array([0,0,0]), (self.login_satellite_angle, self.login_satellite_angle * 0.7, self.login_satellite_angle * 0.4), scale_factor=5.0)

        title_font = pygame.font.SysFont('Arial', 50, bold=True)
        title_text = title_font.render("CubeSat Simulator", True, (0, 200, 255))
        self.screen.blit(title_text, title_text.get_rect(center=(self.screen_width/2, self.screen_height/2 - 200)))

        # Username and Password Fields
        fields = {'username': self.username_input, 'password': self.password_input}
        is_hidden = {'username': self.show_username, 'password': self.show_password}
        
        for i, (field, value) in enumerate(fields.items()):
            field_rect = self.rects[f'{field}_field']
            label = self.draw_text(f"{field.capitalize()}:", (0,0))
            self.screen.blit(label, (field_rect.x - 100, field_rect.y + 5))
            pygame.draw.rect(self.screen, (100, 100, 100), field_rect, 2)
            
            display_text_val = '*' * len(value) if is_hidden[field] else value
            text_surf = self.draw_text(display_text_val, (0,0))
            self.screen.blit(text_surf, (field_rect.x + 5, field_rect.y + 5))
            
            self.draw_eye_icon(f'{field}_toggle', is_hidden[field])

            if self.active_input == field:
                pygame.draw.rect(self.screen, (0, 200, 255), field_rect, 2)
                if self.cursor_visible:
                    cursor_x = field_rect.x + 5 + text_surf.get_width()
                    pygame.draw.line(self.screen, (255,255,255), (cursor_x, field_rect.y + 5), (cursor_x, field_rect.y + 25), 2)
        
        login_rect = self.rects['login_button']
        pygame.draw.rect(self.screen, (0, 150, 0), login_rect)
        login_text = self.draw_text("Login", (0,0), font=self.speed_font)
        self.screen.blit(login_text, login_text.get_rect(center=login_rect.center))

        if self.login_error_message:
            error_text = self.draw_text(self.login_error_message, (0,0), color=(255, 50, 50))
            self.screen.blit(error_text, error_text.get_rect(center=(self.screen_width/2, self.screen_height/2 + 95)))

    def draw_loading_screen(self):
        """Draws a loading bar."""
        self.screen.fill(self.colors['background'])
        loading_text = self.speed_font.render("Loading Simulation...", True, self.colors['text'])
        self.screen.blit(loading_text, loading_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2 - 50)))

        bar_width, bar_height = 400, 30
        bar_rect = pygame.Rect((self.screen_width - bar_width) / 2, self.screen_height / 2, bar_width, bar_height)
        pygame.draw.rect(self.screen, (50, 50, 50), bar_rect, 2)
        
        progress_width = bar_width * (self.loading_progress / 100)
        pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(bar_rect.x, bar_rect.y, progress_width, bar_height))

    def draw_ui(self):
        """Draws the main simulation UI."""
        self.screen.fill(self.colors['background'])
        for star in self.stars: pygame.draw.circle(self.screen, self.colors['star'], star, random.randint(1, 2))

        self.draw_planet()

        if self.cube_sat:
            if len(self.sim_history) > 1:
                trail_points = [tuple(self.view_center + (e['position'][:2] * self.orbital_scale)) for e in self.sim_history]
                pygame.draw.lines(self.screen, (255, 255, 0), False, trail_points, 1)

            if self.is_satellite_visible: self.draw_cube(self.cube_sat.position, self.cube_sat.attitude)
        
        if self.show_history: self.draw_scrollable_panel('telemetry', "Telemetry Data", (0, 255, 0), self.sim_history, self.telemetry_line_generator)
        if self.show_off_attitude_history: self.draw_scrollable_panel('off_attitude', "Off-Attitude Warnings", (255, 150, 50), self.off_attitude_events, self.off_attitude_line_generator)
            
        self.draw_attitude_graphs()

        buttons = {'start_resume_button': "Start/Resume", 'pause_button': "Pause", 'stop_button': "Stop", 'reset_button': "Reset",
                   'toggle_history_button': "Toggle History", 'toggle_satellite_button': "Toggle Sat", 
                   'toggle_off_attitude_button': "Toggle Warnings", 'change_planet_button': "Change Planet"}
        for key, text in buttons.items():
            rect = self.rects[key]
            color = (150,0,0) if key == 'reset_button' else self.colors['button_bg']
            pygame.draw.rect(self.screen, color, rect)
            text_surf = self.draw_text(text, (0,0))
            self.screen.blit(text_surf, text_surf.get_rect(center=rect.center))

        speed_text = self.speed_font.render(f"Speed: {self.simulation_speed_factor:.1f}x", True, self.colors['text'])
        self.screen.blit(speed_text, (20, 20))
        for key in ['slow_down_button', 'speed_up_button']:
            pygame.draw.rect(self.screen, self.colors['button_bg'], self.rects[key])
        self.draw_speed_icons('slow_down_button', "rewind")
        self.draw_speed_icons('speed_up_button', "fast_forward")
        
        if self.cube_sat:
            status_color = (0,255,0) if self.stability_status == "Stable" else (255,255,0)
            if self.is_stabilizing: status_color = (0, 150, 255)
            status_string = f"Status: {'Stabilizing' if self.is_stabilizing else self.stability_status}"
            stability_text = self.speed_font.render(status_string, True, status_color)
            self.screen.blit(stability_text, stability_text.get_rect(center=(self.screen_width / 2, 30)))

            if self.stability_status == "Unstable" and not self.is_stabilizing and self.manual_disturbance_applied:
                rect = self.rects['stabilize_button']
                pygame.draw.rect(self.screen, (0, 200, 200), rect)
                stabilize_text = self.draw_text("Auto-Stabilize", (0,0))
                self.screen.blit(stabilize_text, stabilize_text.get_rect(center=rect.center))

        if self.warning_message:
            warning_text = self.speed_font.render(self.warning_message, True, (255, 0, 0))
            self.screen.blit(warning_text, warning_text.get_rect(center=(self.screen_width / 2, 50)))
            
        if self.show_planet_menu: self.draw_planet_menu()
        if self.show_planet_details: self.draw_planet_details_box()

    def handle_events(self):
        """Processes all user inputs and events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.screen_width, self.screen_height = event.size
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
                self.update_ui_elements()
                self.stars = self.generate_stars()
            
            if self.state == 'LOGIN': self.handle_login_events(event)
            elif self.state == 'SIMULATION': self.handle_simulation_events(event)

    def handle_login_events(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rects['username_field'].collidepoint(event.pos): self.active_input = 'username'
            elif self.rects['password_field'].collidepoint(event.pos): self.active_input = 'password'
            elif self.rects['login_button'].collidepoint(event.pos):
                if self.username_input == "Dinith" and self.password_input == "2002":
                    self.show_loading_bar, self.loading_start_time, self.login_error_message = True, time.time(), ''
                else: self.login_error_message = "Invalid credentials"
            elif self.rects['username_toggle'].collidepoint(event.pos): self.show_username = not self.show_username
            elif self.rects['password_toggle'].collidepoint(event.pos): self.show_password = not self.show_password
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if self.active_input == 'username': self.active_input = 'password'
                else: # on password field
                    if self.username_input == "Dinith" and self.password_input == "2002":
                        self.show_loading_bar, self.loading_start_time, self.login_error_message = True, time.time(), ''
                    else: self.login_error_message = "Invalid credentials"
            elif event.key == pygame.K_TAB: self.active_input = 'password' if self.active_input == 'username' else 'username'
            elif event.key == pygame.K_BACKSPACE:
                if self.active_input == 'username': self.username_input = self.username_input[:-1]
                else: self.password_input = self.password_input[:-1]
            else:
                if self.active_input == 'username': self.username_input += event.unicode
                else: self.password_input += event.unicode

    def handle_simulation_events(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Planet menu/details logic
            if self.show_planet_menu:
                if not self.rects['planet_menu'].collidepoint(event.pos): self.show_planet_menu = False
                for i, planet in enumerate(self.PLANET_MODELS):
                    if self.rects[f"planet_menu_item_{i}"].collidepoint(event.pos):
                        self.selected_planet_for_details = planet
                        self.show_planet_details, self.show_planet_menu = True, False
                        break
            elif self.show_planet_details:
                if self.rects['select_planet_button'].collidepoint(event.pos):
                    self.current_planet_index = self.PLANET_MODELS.index(self.selected_planet_for_details)
                    self.update_simulation_params()
                    self.show_planet_details = False
                elif not self.rects['planet_details'].collidepoint(event.pos) or self.rects['close_details_button'].collidepoint(event.pos):
                    self.show_planet_details = False
            else: # Main simulation buttons
                for key, rect in self.rects.items():
                    if rect.collidepoint(event.pos):
                        self.handle_button_click(key)
                        break
        
        elif event.type == pygame.MOUSEWHEEL:
            if self.show_history and self.rects['telemetry'].collidepoint(pygame.mouse.get_pos()):
                self.history_scroll_offset = max(0, min(self.history_scroll_offset - event.y * 20, max(0, len(self.sim_history) * 54 - self.rects['telemetry'].height + 40)))
            elif self.show_off_attitude_history and self.rects['off_attitude'].collidepoint(pygame.mouse.get_pos()):
                self.off_attitude_scroll_offset = max(0, min(self.off_attitude_scroll_offset - event.y * 20, max(0, len(self.off_attitude_events) * 25 - self.rects['off_attitude'].height + 40)))

        elif event.type == pygame.KEYDOWN and self.cube_sat:
            key_map = {pygame.K_UP: (1, -0.01), pygame.K_DOWN: (1, 0.01), pygame.K_LEFT: (0, -0.01), pygame.K_RIGHT: (0, 0.01)}
            if event.key in key_map:
                axis, delta = key_map[event.key]
                self.cube_sat.attitude[axis] += delta
                self.manual_disturbance_applied = True
    
    def handle_button_click(self, key):
        if key == 'start_resume_button' and self.simulation_state in ['STOPPED', 'PAUSED']:
            self.start_simulation() if self.simulation_state == 'STOPPED' else setattr(self, 'simulation_state', 'RUNNING')
        elif key == 'pause_button' and self.simulation_state == 'RUNNING': self.simulation_state = 'PAUSED'
        elif key == 'stop_button': self.simulation_state = 'STOPPED'
        elif key == 'reset_button': self.reset_simulation()
        elif key == 'slow_down_button': self.simulation_speed_factor = max(0.1, self.simulation_speed_factor / 2.0)
        elif key == 'speed_up_button': self.simulation_speed_factor *= 2.0
        elif key == 'toggle_history_button': self.show_history = not self.show_history
        elif key == 'toggle_satellite_button': self.is_satellite_visible = not self.is_satellite_visible
        elif key == 'toggle_off_attitude_button': self.show_off_attitude_history = not self.show_off_attitude_history
        elif key == 'change_planet_button': self.show_planet_menu = not self.show_planet_menu
        elif key == 'stabilize_button' and self.stability_status == "Unstable":
            self.is_stabilizing = True
            self.manual_disturbance_applied = False
            # Correct orbital velocity
            pos = self.cube_sat.position
            r = np.linalg.norm(pos)
            v_stable_mag = math.sqrt(self.G * self.planet_mass / r)
            tangent_dir = np.array([-pos[1], pos[0], pos[2]])
            self.cube_sat.velocity = (tangent_dir / np.linalg.norm(tangent_dir)) * v_stable_mag


    def update(self):
        """Updates the simulation state."""
        if self.simulation_state != 'RUNNING': return

        dt = self.clock.get_time() / 1000.0 * self.simulation_speed_factor
        self.simulation_time += dt

        if self.cube_sat:
            if self.is_stabilizing:
                self.cube_sat.angular_velocity *= 0.95
                self.cube_sat.attitude *= 0.95
                if np.all(np.abs(np.degrees(self.cube_sat.attitude)) < 0.1) and np.all(np.abs(self.cube_sat.angular_velocity) < 0.01):
                    self.is_stabilizing = False
                    self.cube_sat.attitude, self.cube_sat.angular_velocity = np.zeros(3), np.zeros(3)
            else:
                control_torque = self.attitude_controller.calculate_torque(self.cube_sat.attitude)
                self.cube_sat.update_attitude(control_torque, dt)

            gravity_force = self.calculate_gravity(self.cube_sat.position)
            thrust_force = self.orbital_controller.calculate_thrust(self.cube_sat.position)
            self.cube_sat.update_orbit(gravity_force + thrust_force, dt)

            self.sim_history.append({'time': self.simulation_time, 'position': self.cube_sat.position.copy(),
                                     'velocity': self.cube_sat.velocity.copy(), 'attitude': self.cube_sat.attitude.copy()})
            
            self.stability_status = "Stable" if np.all(np.abs(np.degrees(self.cube_sat.attitude)) < 2.0) else "Unstable"

            if np.any(np.abs(np.degrees(self.cube_sat.attitude)) > 5.0):
                if not self.off_attitude_events or (self.simulation_time - self.off_attitude_events[-1]['time'] > 1.0):
                    self.off_attitude_events.append({'time': self.simulation_time, 'attitude': self.cube_sat.attitude.copy()})
            
            current_radius = np.linalg.norm(self.cube_sat.position)
            if current_radius < self.planet_radius:
                self.warning_message, self.simulation_state = "CRITICAL: Satellite has crashed!", 'STOPPED'
            elif current_radius > self.target_orbit_radius * 5:
                self.warning_message, self.simulation_state = "WARNING: Satellite has escaped stable orbit!", 'STOPPED'

    def run(self):
        """Main application loop."""
        while self.running:
            self.handle_events()
            
            if self.state == 'LOGIN':
                if time.time() - self.last_cursor_toggle > self.cursor_blink_interval:
                    self.cursor_visible = not self.cursor_visible
                    self.last_cursor_toggle = pygame.time.get_ticks()

                self.login_satellite_angle += 0.01
                if self.show_loading_bar:
                    elapsed = time.time() - self.loading_start_time
                    self.loading_progress = min(100, (elapsed / self.loading_duration) * 100)
                    if self.loading_progress >= 100:
                        self.state, self.show_loading_bar = 'SIMULATION', False
                    self.draw_loading_screen()
                else: self.draw_login_screen()
            elif self.state == 'SIMULATION':
                self.update()
                self.draw_ui()

            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

if __name__ == '__main__':
    simulator = CubeSatSimulator()
    simulator.run()

