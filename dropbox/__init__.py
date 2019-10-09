import dropbox
import os
import logging
import cv2
import json
import time
import os

logger = logging.getLogger('keras_mask_rcnn')
logger.setLevel(logging.DEBUG)

class FolderSyncer():
    def read_config(self, config_file):
        """ read DIR from the config """
        with open(config_file) as json_file:
            config = json.load(json_file)
            self.dir = config["ScreenshotsPath"]

    def __init__(self, config_file, authtoken, folder):
        self.read_config(config_file)
        self.authtoken = authtoken
        self.folder = folder
        try:
            self.connect()
        except Exception as ex:
            logger.error("connect() error: %s" % ex)
    
    def connect(self):
        self.dbx = dropbox.Dropbox(self.authtoken)
        user = self.dbx.users_get_current_account()
        logger.info('user: %s' % user)

    def disconnect(self):
        self.dbx = None

    def sync_folder(self):
        for filename in os.listdir(self.dir):
            if filename.endswith(".jpg"):
                # only sync JPG files
                if self.dbx is None:
                    try:
                        self.connect()
                    except Exception as ex:
                        logger.error("connect() error: %s" % ex)
                        return # not connected and cannot connect
                try:
                    self.sync_file(os.path.join(self.dir, filename), filename)
                except Exception as ex:
                    # abort this sync cycle, close connection and force reconnection for next one
                    logger.error("sync_file() error: %s" % ex)
                    self.disconnect()
                    return

    def sync_file(self, file, fname):
        remote_file = self.folder + "/" + fname
        logger.debug("syncing %s into %s" % (file, remote_file))

        img = cv2.imread(file)
        img = cv2.rotate(img, rotateCode=cv2.ROTATE_90_CLOCKWISE)
        res, jpg_img = cv2.imencode('.jpg', img)
        response = self.dbx.files_upload(jpg_img.tobytes(), remote_file, mute=True)
        logger.debug('uploaded: %s' % response)

        os.remove(file)


TEST_AUTHTOKEN="nuL9MZ6SjMAAAAAAAAAAFIkcXRWlmzknzhbSncXo8Ove5TsjLd0MrQ0n7sn79QC7"
TEST_CONFIG="settings.json"
TEST_FOLDER="/LoveArRobots"

if __name__ == "__main__":
    logging.basicConfig()
    logging.basicConfig(level=logging.ERROR, format='%(message)s')

    fs = FolderSyncer(TEST_CONFIG, TEST_AUTHTOKEN, TEST_FOLDER)
    while True:
        fs.sync_folder()
        time.sleep(1)
