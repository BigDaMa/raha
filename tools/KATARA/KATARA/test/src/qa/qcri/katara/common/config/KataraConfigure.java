package qa.qcri.katara.common.config;

import java.io.IOException;
import java.io.PrintStream;
import java.util.Properties;
import java.util.TreeSet;

import jline.console.ConsoleReader;

public class KataraConfigure {

	public static Properties config = new Properties();

	// build version of katara project
	public static String BUILD_VERSION;
	// local database connection
	public static String DATABASE_CONNECTION_URL;
	// local database authentication username
	public static String DATABASE_USERNAME;
	// local database authentication password
	public static String DATABASE_PASSWORD;

	public synchronized static void loadConfig() {
		if (config.isEmpty()) {
			try {
				config.load(KataraConfigure.class.getClassLoader()
						.getResourceAsStream("kataraconfigure.properties"));
			} catch (IOException exc) {
				exc.printStackTrace();
			}
		}
	}

	static {
		loadConfig();
		BUILD_VERSION = config.getProperty("buildversion");
		DATABASE_CONNECTION_URL = config.getProperty("database.connectionurl");
		DATABASE_USERNAME = config.getProperty("database.username");
		DATABASE_PASSWORD = config.getProperty("database.password");

	}
	
	
	public static void setDataBaseConnectionURL(String databaseConnectionURL) {
		DATABASE_CONNECTION_URL = databaseConnectionURL;
	}
	
	public static void setDataBaseUserName(String userName) {
		DATABASE_USERNAME = userName;
	}
	
	public static void setDataBasePassword(String password) {
		DATABASE_PASSWORD = password;
	}
	

	/**
	 * print out settings, this function is used for diagnose and debug purpose
	 * 
	 * @param stream
	 */
	public static void show(PrintStream stream) {
		for (Object key : new TreeSet<Object>(config.keySet())) {
			stream.println(key + "=" + config.getProperty(key.toString()));
		}
	}

	/**
	 * print out settings, this function is used for diagnose and debug purpose
	 * 
	 * @param stream
	 */
	public static void show(ConsoleReader stream) {
		try {
			for (Object key : new TreeSet<Object>(config.keySet())) {
				stream.println(key + "=" + config.getProperty(key.toString()));
			}
			stream.flush();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	public static class KataraCrowdClient {

		static {
			loadConfig();
			// katara crowd client configure
			KataraCrowdClient.CROWD_SERVER_HOSTNAME = config
					.getProperty("crowdclient.hostname");
			}
		public static String CROWD_SERVER_HOSTNAME;
	}

	public static class KataraPatternDiscovery {

		static {
			loadConfig();
			// Katara pattern discovery configuration
			KataraPatternDiscovery.BAYES_ISALUCENE_DIRECTORY_PATH = config
					.getProperty("patterndiscovery.bayes.isalucenedirectorypath");
			KataraPatternDiscovery.BAYES_RELATIONLUCENE_DIRECTORY_PATH = config
					.getProperty("patterndiscovery.bayes.relationlucenedirectorypath");
			KataraPatternDiscovery.BAYES_FUZZYSCORE_THREHOLD = Float
					.parseFloat(config
							.getProperty("patterndiscovery.bayes.fuzzyscorethrehold"));
			KataraPatternDiscovery.RANKJOIN_INTERESTEDTYPEPATH = config
					.getProperty("patterndiscovery.rankjoin.interestedtypepath");
			KataraPatternDiscovery.RANKJOIN_TYPERELCOHERENCEPATH = config
					.getProperty("patterndiscovery.rankjoin.typerelcoherencepath");
		}
		public static String BAYES_ISALUCENE_DIRECTORY_PATH;

		public static String BAYES_RELATIONLUCENE_DIRECTORY_PATH;

		public static float BAYES_FUZZYSCORE_THREHOLD;

		public static String RANKJOIN_TYPERELCOHERENCEPATH;

		public static String RANKJOIN_INTERESTEDTYPEPATH;

	}

	public static class Crowd{
		static {
			loadConfig();
			Crowd.CROWD_HIT_NO = Integer.parseInt(config.getProperty("crowd.hitno"));
		}
		
		public static Integer CROWD_HIT_NO;
	}
}
