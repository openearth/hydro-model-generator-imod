import sys
import click
import logging
import hydro_model_generator_imod
from hydro_model_builder import model_builder

logger = logging.getLogger(__name__)


@click.group()
def main():
    return 0


@click.command(name="generate-model")
@click.option("-o", "--options-file", required=True, help="Options file in YAML format")
@click.option("-r", "--results-dir", required=True, help="Result directory")
@click.option(
    "--skip-download/--no-skip-download",
    default=False,
    help="Skip downloading data if already done",
)
def generate_model(options_file, results_dir, skip_download):
    # two YAML docs are expected in this file, one generic and one model specific
    general_options, model_options = model_builder.parse_config(options_file)
    msg = f"Going to create an imodflow model, it will be placed in '{results_dir}'"
    print(msg)
    if not skip_download:
        model_builder.fetch_data(general_options)

    hydro_model_generator_imod.model_generator.build_model(**model_options, general_options=general_options)


main.add_command(generate_model)

if __name__ == "__main__":
    # use sys.argv[1:] to allow using PyCharm debugger
    # https://github.com/pallets/click/issues/536
    main(sys.argv[1:])  # pragma: no cover
