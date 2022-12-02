def gene_from_fname(fname):
    params = fname.name.split('_')
    chosen = None
    for p in params:
        try:
            int(p)
        except ValueError:
            chosen = p
            break
    return chosen
