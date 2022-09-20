import train
import test

output_path, only = train.run()
test.run(output_path, only)
