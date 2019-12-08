import click, glob, json, os
from neurpy.config import initialize
from neurpy.callback import Callback

@click.command()
@click.argument('config', default='config/stylegan_1024x1024_v1.json')
def run(config):
    if os.path.isfile(config): configs = [config]
    else: configs = glob.glob(os.path.join(config,'*.json'))

    for config in configs:
        with open(config, 'r') as f: args = json.load(f)
        model, datagen, cfg = initialize(args)
        print(cfg.experiment)

        callback = Callback(model=model,
                            experiment=cfg.experiment,
                            saver=cfg.saver,
                            save_interval=cfg.save_interval,
                            test_interval=cfg.test_interval)

        model.compile(cfg)

        model.train(cfg, datagen, callback=callback)

if __name__ == '__main__':
    run()
