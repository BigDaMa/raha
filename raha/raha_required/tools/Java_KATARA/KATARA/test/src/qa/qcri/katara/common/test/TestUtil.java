package qa.qcri.katara.common.test;

import java.io.IOException;
import java.util.Properties;

public class TestUtil {

	public static Properties config = new Properties();

	private static String KDB_PATH;

	private static String OUTPUT_DIRECTORY;

	public static String getKDBPath() {
		return KDB_PATH;
	}

	public static String getOutputDirectory() {
		return OUTPUT_DIRECTORY;
	}

	public synchronized static void loadConfig() {
		try {
			config.load(TestUtil.class.getClassLoader().getResourceAsStream(
					"test.properties"));
		} catch (IOException exc) {
			exc.printStackTrace();
		}
	}

	static {
		loadConfig();
		KDB_PATH = config.getProperty("kdbpath");
		OUTPUT_DIRECTORY = config.getProperty("outputdirectory");
	}

}
