package qa.qcri.katara.common;

import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;

public class JarClassLoader extends URLClassLoader {
	
	public JarClassLoader(URL[] urls) {
		super(urls);
	}
	
	public void addJarFile(String path) throws MalformedURLException{
		String urlPath = "file:///"+path;
		addURL(new URL(urlPath));
	}
}
