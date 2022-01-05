from senne.senne import Ensembler

if __name__ == '__main__':
    # An example of how to launch ensembling algorithm for already trained neural networks
    ensembler = Ensembler(path='example_folder', device='cuda')

    # Create an ensemble
    ensembler.prepare_composite_model(final_model='ridge')
