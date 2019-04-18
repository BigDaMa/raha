package qa.qcri.katara.common;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class FileUtil {

	public static void refreshFile(String filePath) throws IOException {
		File file = new File(filePath);
		if (file.exists()) {
			boolean isDeleted = file.delete();
			if (isDeleted == false) {
				throw new IOException("file:" + filePath + " is being occupied");
			}
		}
	}

	public static void copy(String sourceFilePath, String targetFilePath)
			throws IOException {
		InputStream input = null;
		OutputStream output = null;
		try {
			input = new FileInputStream(sourceFilePath);
			output = new FileOutputStream(targetFilePath);
			byte[] buf = new byte[1024];
			int bytesRead;
			while ((bytesRead = input.read(buf)) > 0) {
				output.write(buf, 0, bytesRead);
			}
		} finally {
			input.close();
			output.close();
		}
	}

	public static void clearDirectory (String directoryPath, boolean remove) {
		File directory = new File(directoryPath);
		File[] files = directory.listFiles();
		if (files != null) { // some JVMs return null for empty dirs
			for (File f : files) {
				if (f.isDirectory()) {
					clearDirectory(f.getAbsolutePath(), true);
				} else {
					f.delete();
				}
			}
		}
		if (remove == true)
			directory.delete();
	}

	public static void mkdir(String directoryPath) throws IOException {
		boolean isSuccess = new File(directoryPath).mkdirs();
		if (!isSuccess) {
			throw new IOException("unable to create directory in "
					+ directoryPath);
		}
	}

}
