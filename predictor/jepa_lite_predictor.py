from predictor.base_predictor import BasePredictor


class JEPALitePredictor(BasePredictor):

    def predict_next_state(self, z_t, skill_spec):

        agent_r, agent_c = z_t["agent_pos"]
        goal_r, goal_c = z_t["goal_pos"]

        walls = z_t["local_walls"].copy()

        skill = skill_spec["skill"]

        if skill != "move":
            new_r, new_c = agent_r, agent_c

        else:
            direction = skill_spec["args"]["direction"].lower()

            if direction == "up" and not walls["up"]:
                new_r, new_c = agent_r - 1, agent_c
            elif direction == "down" and not walls["down"]:
                new_r, new_c = agent_r + 1, agent_c
            elif direction == "left" and not walls["left"]:
                new_r, new_c = agent_r, agent_c - 1
            elif direction == "right" and not walls["right"]:
                new_r, new_c = agent_r, agent_c + 1
            else:
                new_r, new_c = agent_r, agent_c

        dx = goal_c - new_c
        dy = goal_r - new_r

        predicted = {
            "agent_pos": (new_r, new_c),
            "goal_pos": (goal_r, goal_c),
            "dx": dx,
            "dy": dy,
            "goal_distance": abs(dx) + abs(dy),
            "local_walls": {
                "up": None,
                "down": None,
                "left": None,
                "right": None,
            },
            "step_count": z_t["step_count"] + 1,
        }

        return predicted
