from manim import *
import numpy as np

class DykstraVisualization(Scene):
    def construct(self):
        # 1. Configuration and Setup
        self.camera.background_color = "#ffffff"
        
        # Define two convex sets (Circles for simplicity)
        # Set A: Left Circle
        center_a = np.array([-1.5, -0.5, 0])
        radius_a = 2.0
        circle_a = Circle(radius=radius_a, color=BLUE, fill_opacity=0.2).move_to(center_a)
        label_a = MathTex("C_1", color=BLUE).next_to(circle_a, UL)

        # Set B: Right Circle
        center_b = np.array([1.5, -0.5, 0])
        radius_b = 2.0
        circle_b = Circle(radius=radius_b, color=RED, fill_opacity=0.2).move_to(center_b)
        label_b = MathTex("C_2", color=RED).next_to(circle_b, UR)

        # Draw the sets
        self.play(Create(circle_a), Write(label_a), run_time=1)
        self.play(Create(circle_b), Write(label_b), run_time=1)
        
        # 2. Algorithm Initialization
        # Start point (outside both)
        start_point = np.array([0, 3.0, 0])
        dot_start = Dot(start_point, color=YELLOW, radius=0.1)
        label_start = MathTex("x_0", color=YELLOW).next_to(dot_start, UP)

        self.play(FadeIn(dot_start), Write(label_start))

        # Variables for Dykstra's Algorithm
        # x: current estimate
        # p: increment for Set A (correction vector)
        # q: increment for Set B (correction vector)
        
        x = start_point
        p = np.zeros(3)
        q = np.zeros(3)
        
        # Projection Helper Function for Circle
        def project_to_circle(point, center, radius):
            vec = point - center
            dist = np.linalg.norm(vec)
            if dist <= radius:
                return point # Already inside
            return center + (vec / dist) * radius

        # Track graphical elements to fade them out later
        previous_elements = VGroup()
        
        # 3. Animation Loop
        iterations = 5
        path_trace = VGroup()
        path_trace.add(dot_start)

        # Normal speed for viewing
        anim_speed = 0.7

        for k in range(iterations):
            # --- Step 1: Project onto C1 ---
            # Input to projection is (x + p)
            input_a = x + p
            projected_a = project_to_circle(input_a, center_a, radius_a)
            
            # Update increment p
            new_p = input_a - projected_a
            
            # Visualization for Step 1
            step_text = MathTex(f"k={k+1}: P_{{C_1}}(x + p)", font_size=24).to_corner(UL)
            
            # Show the ghost point input (x + p) if p is non-zero
            ghost_dot_a = Dot(input_a, color=BLUE_A, fill_opacity=0.5)
            ghost_line_a = DashedLine(x, input_a, color=BLUE_A)
            
            if k > 0: # Only interesting after first iteration
                self.add(step_text, ghost_dot_a, ghost_line_a)
                self.wait(0.3)
            else:
                self.add(step_text)

            # Animate projection to A
            dot_a = Dot(projected_a, color=BLUE)
            proj_line_a = Line(input_a, projected_a, color=BLUE)
            
            self.play(
                Create(proj_line_a),
                TransformFromCopy(ghost_dot_a if k > 0 else Dot(x, radius=0), dot_a),
                run_time=anim_speed
            )
            
            # Update variables
            y = projected_a # Temporary intermediate point
            p = new_p

            # Cleanup visuals
            previous_elements.add(ghost_dot_a, ghost_line_a, proj_line_a, step_text)
            if k > 0: self.remove(ghost_dot_a, ghost_line_a, step_text) # Keep trace clean

            # --- Step 2: Project onto C2 ---
            # Input to projection is (y + q)
            input_b = y + q
            projected_b = project_to_circle(input_b, center_b, radius_b)
            
            # Update increment q
            new_q = input_b - projected_b
            
            # Visualization for Step 2
            step_text_b = MathTex(f"k={k+1}: P_{{C_2}}(y + q)", font_size=24).to_corner(UL)
            
            ghost_dot_b = Dot(input_b, color=RED_A, fill_opacity=0.5)
            ghost_line_b = DashedLine(y, input_b, color=RED_A)
            
            if k > 0: # q is zero initially, so only show ghost after first loop
                self.add(step_text_b, ghost_dot_b, ghost_line_b)
                self.wait(0.3)
            else:
                self.add(step_text_b)

            # Animate projection to B
            dot_b = Dot(projected_b, color=RED)
            proj_line_b = Line(input_b, projected_b, color=RED)
            
            self.play(
                Create(proj_line_b),
                TransformFromCopy(ghost_dot_b if k > 0 else Dot(y, radius=0), dot_b),
                run_time=anim_speed
            )

            # Update variables for next loop
            x = projected_b
            q = new_q
            
            # Clean up loop visuals
            previous_elements.add(ghost_dot_b, ghost_line_b, proj_line_b, step_text_b, dot_a)
            self.remove(step_text, step_text_b)
            
            # Fade out construction lines to reduce clutter, keep the points
            self.play(
                FadeOut(ghost_line_a), FadeOut(ghost_line_b),
                FadeOut(proj_line_a), FadeOut(proj_line_b),
                run_time=0.2
            )
            
            path_trace.add(dot_a, dot_b)

        # 4. Final Result
        final_dot = Dot(x, color=GREEN, radius=0.15)
        final_label = MathTex("P_{C_1 \cap C_2}(x_0)", color=GREEN).next_to(final_dot, DOWN)
        
        self.play(Transform(dot_b, final_dot), Write(final_label), run_time=1)
        
        # Draw the true intersection line segment
        intersection_segment = Line([0, -1.32, 0], [0, 1.32, 0], color=GREEN, stroke_opacity=0.5)
        self.play(Create(intersection_segment))
        
        conclusion = Text("Converged", font_size=30).to_corner(DR)
        self.play(Write(conclusion))
        self.wait(2)

if __name__ == "__main__":
    # Config settings to make it behave like "manim -pql ..."
    config.pixel_height = 720
    config.pixel_width = 1280
    config.frame_height = 8.0
    config.frame_width = 8.0 * (1280 / 720)
    config.background_color = "#1e1e1e"    
    scene = DykstraVisualization()
    scene.render()