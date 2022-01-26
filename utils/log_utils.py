import sys
import logging
import datetime


class LogUtils:

    @staticmethod
    def log_config():
        time_stamp = datetime.datetime.now().strftime("%d-%b-%Y-(%H.%M.%S.%f)")
        logging_filename = '{}'.format(time_stamp)
        stdout_handler = logging.StreamHandler(sys.stdout)

        logging.basicConfig(filename=r'..\Results\logs\{}.txt'.format(logging_filename), level=logging.DEBUG,
                            format='%(levelname)s %(asctime)s %(message)s')

        # make logger print to console (it will not if multithreaded)
        logging.getLogger().addHandler(stdout_handler)

