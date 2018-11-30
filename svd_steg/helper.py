"""Helper File."""


def progress_bar(current_iteration, total_iterations,
                 decimals=2, length=100, fill=u"\u2588"):
    """Print a progress bar."""
    percent = ("{0:." + str(decimals)
               + "f}").format(100*(current_iteration/float(total_iterations)))
    filled_length = int(length * current_iteration // total_iterations)
    prog_bar = fill * filled_length + '_' * (length - filled_length)
    print('\rProgress: |%s %s%%' % (prog_bar, percent), end='\r')
    if current_iteration == total_iterations:
        print()
