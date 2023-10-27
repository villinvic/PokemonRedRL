from typing import List

from pyboy.pyboy import PyBoy


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
    return [
        read_m(console, addr)
        for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
    ]

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

if __name__ == '__main__':
    console = PyBoy(
        "PokemonRed.gb",
        debugging=False,
        disable_input=False,
        window_type='headless',
        hide_window=True,
    )

    screen = console.botsupport_manager().screen()
    console.set_emulation_speed(0)

    with open("has_pokedex_nballs.state", "rb") as f:
        console.load_state(f)


    while True:
        console.tick()

        print(
            read_party_ptypes(console),
            read_party_health(console),
            read_party_levels(console),
            read_party_experience(console))
        input()