from time import time
from typing import List

import numpy as np
from pyboy.pyboy import PyBoy
from pyboy.utils import WindowEvent


def read_m(console, addr):
    return console.get_memory_value(addr)

def read_double(console, start_addr):
    return 256 * read_m(console, start_addr) + read_m(console, start_addr + 1)

def read_triple(console, start_addr):
    return (
            256 * 256 * read_m(console, start_addr)
            + 256 * read_m(console, start_addr + 1)
            + read_m(console, start_addr + 2)
    )

def read_party_ptypes(console):

    return console.get_memory_batch(0xD164, 0xD169)

def read_party_experience(console) -> List:
    return [
        read_triple(console, addr)
        for addr in [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]
    ]

def read_party_levels(console) -> List:

    return [
        read_m(console, addr)
        for addr in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
    ]

def read_party_health(console) -> List:
    """
    :return: current party health fractions
    """

    return [
        read_double(console, curr_hp_addr)/(read_double(console, max_hp_addr)+1e-8)
        for curr_hp_addr, max_hp_addr in
        [(0xD16C, 0xD18D), (0xD198, 0xD1B9), (0xD1C4, 0xD1E5), (0xD1F0, 0xD211), (0xD21C, 0xD23D), (0xD248, 0xD269)]
    ]

def read_caught_raw(console) -> int:

    total_caught = [
        bin(read_m(console, addr)) for addr in range(0xD2F7, 0xD309)
    ]
    return total_caught

def read_seen_raw(console) -> int:

    total_seen = [
        bin(read_m(console, addr)) for addr in range(0xD30A, 0xD31C)
    ]
    return total_seen

def read_pos(console) -> List:
    return [read_m(console, 0xD362), read_m(console, 0xD361)]

def read_map(console) -> List:
    return read_m(console, 0xD35E)

def read_walking_animation(console):
    return read_m(console, 0xC108)

def read_opp_level(console):
    return read_m(console, 0xCFF3)

def read_extensive_events(console) -> List[int]:
    t = time()
    binary_components = []

    for i in range(0xD747, 0xD886):
        value = read_m(console, i)
        binary_components.extend([int(bit) for bit in format(value, '08b')])

    print("time:", time()-t)
    print(len(binary_components))

    return binary_components

def find_ones_indices(console) -> List[int]:
    ones_indices = []
    t = time()

    for i in range(0xD747, 0xD886):
        value = read_m(console, i)

        index = 0
        while value:
            if value & 1:
                ones_indices.append((i-0xD747) * 8 + index)
            value >>= 1
            index += 1

    print("time2 :", time()-t)

    return ones_indices

def read_sent_out(console) -> List:
    return [int(i == read_m(console, 0xCC2F)) for i in range(6)]

def read_bcd(console, addr):
    num = read_m(console, addr)
    return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)
def read_money(console):
    return (100 * 100 * read_bcd(console, 0xD347) +
            100 * read_bcd(console, 0xD348) +
            read_bcd(console, 0xD349))


def read_combat(console) -> int:
    return read_m(console, 0xD057)

def read_turn_count(console) -> int:
    return read_m(console, 0xCCD5)

def read_action_taken(console) -> int:
    return read_m(console, 0xCCF1) # CCF2

def read_textbox_id(console) -> int:
    return read_m(console, 0xD125)

def read_party_mon_menu(console) -> int:
    return read_m(console, 0xd09b)

def read_movement_y(console) -> int:
    return read_m(console, 0xC103)

def read_movement_x(console) -> int:
    return read_m(console, 0xC105)

valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START
]
release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]

action_dict = {
    "[A": 3,
    "[B": 0,
    "[D": 1,
    "[C": 2,
    "": 4,
    "0": 5,
    "5": 6

}

def skip_battle_frame(console):

    if read_combat(console) and not (
            read_textbox_id(console) in {11, 12, 13, 20}
        or
            read_party_mon_menu(console) > 0
    ):
        c = 0
        for i in range(18 * 32):
            # Skip battle animations
            console.tick()
            if not read_combat(console) or (
            read_textbox_id(console) in {11, 12, 13, 20}
        or
            read_party_mon_menu(console) > 0
    ):
                c += 1
                if c == 5:
                    # Needs one more action when battle ends
                    break

def step(console, action):

    # press button then release after some steps
    console.send_input(valid_actions[action])
    walked = False
    for i in range(19):
        # release action, so they are stateless
        if not walked:
            walked = read_walking_animation(console) > 0
        if i == 8:

            if action < 4:
                # release arrow
                console.send_input(release_arrow[action])

            if 3 < action < 6:
                # release button
                console.send_input(release_button[action - 4])
            if valid_actions[action] == WindowEvent.PRESS_BUTTON_START:
                console.send_input(WindowEvent.RELEASE_BUTTON_START)
        if i >= 8:
                skip_battle_frame(console)
                skip_empty_screen(console)

        console.tick()


    return walked


def skippable_screen(console):
    screen = console.botsupport_manager().screen().screen_ndarray()
    grayscale_screen = np.uint8(
        0.299 * screen[:, :, 0]
        + 0.587 * screen[:, :, 1]
        + 0.114 * screen[:, :, 2]
    )
    blackness = np.sum(np.int32(grayscale_screen <= 12)) / (144*160)
    whiteness = np.sum(np.int32(grayscale_screen >= 254)) / (144 * 160)

    print(blackness, whiteness)
    return (
            blackness > 0.75
            or
            whiteness >= 0.99
            )

def skip_empty_screen(console):
    skipped = 0
    while skippable_screen(console):
        skipped += 1
        console.tick()

        if skipped > 1000:
            print("what")

    if skipped > 0:
        for k in range(10):
            console.tick()

if __name__ == '__main__':
    console = PyBoy(
        "pokered.gbc",
        debugging=False,
        disable_input=True,
        hide_window=False,
        disable_renderer=False
    )

    print(valid_actions[5])

    screen = console.botsupport_manager().screen()
    #console.set_emulation_speed(0)
    #console.get_memory_value("bh")

    with open("deepred_post_parcel_pokeballs.state", "rb") as f:
        console.load_state(f)
    console.tick()
    console.tick()

    walked = 0
    past_walked_locations = [(0, 0, -1), (0, 0, -1)]

    def get_allowed_actions():
        allowed_actions = [1] * len(valid_actions)
        x, y = read_pos(console)
        map_id = read_map(console)
        combat = read_combat(console)

        loc = x, y, map_id
        if past_walked_locations[-1] != loc:
            past_walked_locations.append(loc)

        if not combat:
            curr_x, curr_y, _ = past_walked_locations[-1]
            past_x, past_y, _ = past_walked_locations[-2]

            if curr_y - past_y == 1:
                allowed_actions[3] = 0
            elif curr_y - past_y == -1:
                allowed_actions[0] = 0
            elif curr_x - past_x == 1:
                allowed_actions[1] = 0
            elif curr_x - past_x == -1:
                allowed_actions[2] = 0
        return allowed_actions

    while True:
        #print(screen.screen_ndarray())
        print(
            (read_pos(console), read_map(console)),
            read_opp_level(console),
            read_combat(console),
            read_textbox_id(console),
            read_party_mon_menu(console)
        )

        inputs = input("input:").split("\x1b")
        if len(inputs) == 1:
            i = inputs[0]
            if i not in action_dict:
                    i = ""

            allowed_actions = get_allowed_actions()
            if allowed_actions[action_dict[i]] == 0:
                i = ""

            walked = step(console, action_dict[i])
        else:
            for i in inputs[1:]:
                if i not in action_dict:
                    i = ""

                allowed_actions = get_allowed_actions()
                if allowed_actions[action_dict[i]] == 0:
                    i = ""

                walked = step(console, action_dict[i])

