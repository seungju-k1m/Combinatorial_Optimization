from COP_RL.Config import NeuralCOCfg
from COP_RL.Player import Player

import argparse


print("HELLO")

parser = argparse.ArgumentParser(
    description="Combinatorial Optimization with RL.")
parser.add_argument(
    "--path",
    "-p",
    type=str,
    default="./cfg/COP_RL.json",
    help="A path of configuration file."
)

parser.add_argument(
    "--plot",
    "-plt",
    action="store_true",
    default=False,
    help="if you add this tag, plot mode is on."
)

parser.add_argument(
    "--eval",
    "-e",
    action="store_true",
    default=False,
    help="if you add this tag, eval mode is on. \
          you can specify configuration of evaluation in your.json *"
)

parser.add_argument(
    "--log",
    "-l",
    action="store_true",
    default=False,
    help="It is always false if it is not train mode."
)

args = parser.parse_args()

if __name__ == "__main__":
    path = args.path
    cfg = NeuralCOCfg(path)
    if args.eval or args.plot:
        if args.eval:
            player = Player(cfg, logMode=False)
    else:
        player = Player(cfg, logMode=args.log)
        player.run()

    if args.plot:
        Player.plot(cfg)

    if args.eval:
        player.eval()
