import train
import test
from logger import console_logger, file_logger

console_logger()
file_logger()

output_path, only = train.run()
test.run(output_path, only)
