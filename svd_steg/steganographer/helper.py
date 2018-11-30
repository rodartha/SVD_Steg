"""Helper File."""
def progressBar(current_iteration, total_iterations, decimals=2, length=100, fill=u"\u2588"):
	percent = ("{0:." + str(decimals) + "f}").format(100*(current_iteration/float(total_iterations)))
	filled_length = int(length * current_iteration // total_iterations)
	bar = fill * filled_length + '_' * (length - filled_length)
	print('\rProgress: |%s %s%%' % (bar, percent), end = '\r')
	if current_iteration == total_iterations:
		print()
