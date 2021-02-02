progress_plug = ProgressMonitor()
random_plug = RandomMonitor(10000)
image_plug = ConstantMonitor(data.coffee().swapaxes(0,2).swapaxes(1,2), "image")

# Loggers are a special type of plugin which, surprise, logs the stats
logger = Logger(["progress"], [(2, 'iteration')])
text_logger    = VisdomTextLogger(["progress"], [(2, 'iteration')], update_type='APPEND',
                    env=env, opts=dict(title='Example logging'))
scatter_logger = VisdomPlotLogger('scatter', ["progress.samples_used", "progress.percent"], [(1, 'iteration')],
                    env=env, opts=dict(title='Percent Done vs Samples Used'))
hist_logger    = VisdomLogger('histogram', ["random.data"], [(2, 'iteration')],
                    env=env, opts=dict(title='Random!', numbins=20))
image_logger   = VisdomLogger('image', ["image.data"], [(2, 'iteration')], env=env)


# Create a saver
saver = VisdomSaver(envs=[env])

# Register the plugins with the trainer
train.register_plugin(progress_plug)
train.register_plugin(random_plug)
train.register_plugin(image_plug)

train.register_plugin(logger)
train.register_plugin(text_logger)
train.register_plugin(scatter_logger)
train.register_plugin(hist_logger)
train.register_plugin(image_logger)

train.register_plugin(saver)