import dropbox
import os
import logging
import cv2

logger = logging.getLogger('keras_mask_rcnn')
logger.setLevel(logging.DEBUG)

class FolderSyncer():
    def __init__(self, dir, authtoken, folder):
        self.dir = dir
        self.folder = folder
        self.dbx = dropbox.Dropbox(authtoken)
        user = self.dbx.users_get_current_account()
        logger.info('user: %s' % user)

    def sync_folder(self):
        for filename in os.listdir(self.dir):
            if filename.endswith(".jpg"): 
                self.sync_file(os.path.join(self.dir, filename), filename)

    def sync_file(self, file, fname):
        remote_file = self.folder + "/" + fname
        logger.debug("syncing %s into %s" % (file, remote_file))

        img = cv2.imread(file)
        res, jpg_img = cv2.imencode('.jpg', img)
        response = self.dbx.files_upload(jpg_img.tobytes(), remote_file, mute=True)
        logger.debug('uploaded: %s' % response)

TEST_AUTHTOKEN="nuL9MZ6SjMAAAAAAAAAAFIkcXRWlmzknzhbSncXo8Ove5TsjLd0MrQ0n7sn79QC7"
TEST_DIR="/tmp"
TEST_FOLDER="/LoveArRobots"

if __name__ == "__main__":
    logging.basicConfig()
    logging.basicConfig(level=logging.ERROR, format='%(message)s')

    fs = FolderSyncer(TEST_DIR, TEST_AUTHTOKEN, TEST_FOLDER)
    fs.sync_folder()
