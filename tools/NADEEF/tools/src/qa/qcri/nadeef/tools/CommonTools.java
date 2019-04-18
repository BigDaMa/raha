/*
 * QCRI, NADEEF LICENSE
 * NADEEF is an extensible, generalized and easy-to-deploy data cleaning platform built at QCRI.
 * NADEEF means "Clean" in Arabic
 *
 * Copyright (c) 2011-2013, Qatar Foundation for Education, Science and Community Development (on
 * behalf of Qatar Computing Research Institute) having its principle place of business in Doha,
 * Qatar with the registered address P.O box 5825 Doha, Qatar (hereinafter referred to as "QCRI")
 *
 * NADEEF has patent pending nevertheless the following is granted.
 * NADEEF is released under the terms of the MIT License, (http://opensource.org/licenses/MIT).
 */

package qa.qcri.nadeef.tools;

import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.hash.Hashing;

import javax.tools.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.Socket;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.List;

/**
 * Common helper tools.
 */
public final class CommonTools {
    private static Logger tracer = Logger.getLogger(CommonTools.class);
    private static Process derbyProcess;

    public static char DOUBLE_QUOTE = '"';
    public static char SINGLE_QUOTE = '\'';

    public static boolean isWindows() {
        String osName = System.getProperty("os.name");
        return osName.indexOf("Win") >= 0;
    }

    public static boolean isMac() {
        String osName = System.getProperty("os.name");
        return osName.indexOf("Mac") >= 0;
    }

    public static boolean isLinux() {
        String osName = System.getProperty("os.name");
        return osName.indexOf("Linux") >= 0;
    }

    /**
     * Gets the file object given the full file name.
     * @param fileName full file name.
     * @return File object.
     */
    public static File getFile(String fileName) throws FileNotFoundException {
        if (isWindows()) {
            fileName = fileName.replace("\\", "\\\\");
        } else {
            fileName = fileName.replace("\\", "/");
        }

        File file = new File(fileName);
        if (!file.exists()) {
            throw new FileNotFoundException();
        }
        if (!file.isFile()) {
            throw new IllegalArgumentException("The given filename is not a file.");
        }
        return file;
    }

    /**
     * Loads a class in runtime using specified classpath.
     * @param className class name.
     * @param url classpath url.
     * @return Class type.
     */
    public static Class loadClass(String className, URL url) throws ClassNotFoundException {
        URL[] urls = new URL[] { url };
        URLClassLoader ucl = new URLClassLoader(urls);
        return ucl.loadClass(className);
    }

    /**
     * Loads a class in runtime using default classpath.
     * @param className class name.
     * @return Class type.
     */
    public static Class loadClass(String className) throws ClassNotFoundException {
        ClassLoader classLoader = ClassLoader.getSystemClassLoader();
        return classLoader.loadClass(className);
    }

    /**
     * Returns <code>True</code> when the given string is a numerical string.
     * @param string input string.
     * @return <code>True</code> when the given string is a numerical string.
     */
    public static boolean isNumericalString(String string) {
        if (string == null) {
            return false;
        }
        return string.matches("-?\\d+(\\.\\d+)?");
    }

    /**
     * Returns <code>True</code> when the given attribute name is a valid full column name.
     * @param attributeName Column name.
     * @return <code>True</code> when the column name is valid.
     */
    public static boolean isValidColumnName(String attributeName) {
        String[] tokens = attributeName.split("\\.");
        // TODO: change to correct regexp.
        if (
            tokens.length != 2 ||
            Strings.isNullOrEmpty(tokens[0]) ||
            Strings.isNullOrEmpty(tokens[1])
        ) {
            return false;
        }
        return true;
    }

    /**
     * Compiles the input file and generates the output class file.
     * @param fullFilePath input .java file path.
     * @return class file path.
     */
    public static String compileFile(File fullFilePath) throws IOException {
        Preconditions.checkArgument(fullFilePath != null && fullFilePath.isFile());
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        Preconditions.checkNotNull(compiler);
        DiagnosticCollector<JavaFileObject> diagnostics =
            new DiagnosticCollector<JavaFileObject>();
        StandardJavaFileManager fileManager =
            compiler.getStandardFileManager(diagnostics, null, null);
        Iterable<? extends JavaFileObject> compilationUnit =
            fileManager.getJavaFileObjectsFromFiles(Arrays.asList(fullFilePath));
        List<String> args = Arrays.asList("-Xlint:none");
        boolean result =
            compiler.getTask(null, fileManager, diagnostics, args, null, compilationUnit).call();

        String msg = "";
        if (!result) {
            for (Diagnostic diagnostic : diagnostics.getDiagnostics()) {
                msg +=
                    String.format(
                        "Error on line %d in %s%n",
                            diagnostic.getLineNumber(),
                            diagnostic.getMessage(null));
            }
            tracer.error(msg);
        }
        fileManager.close();
        return msg;
    }

    /**
     * Converts a string into an integer hashcode.
     * @param value string value.
     * @return integer hashcode.
     */
    public static int toHashCode(String value) {
        return Math.abs(
            Hashing
                .md5()
                .newHasher()
                .putString(value, Charset.forName("UTF8"))
                .hash()
                .asInt()
        );
    }

    public static boolean isPortOccupied(int port) {
        try (Socket ignored = new Socket("localhost", port)) {
            return true;
        } catch (IOException ignored) {
            return false;
        }
    }

    public static boolean waitForService(int port) {
        final int MAX_TRY_COUNT = 10;
        int tryCount = 0;
        while (true) {
            if (isPortOccupied(port)) {
                return true;
            }

            if (tryCount == MAX_TRY_COUNT) {
                System.err.println("Waiting for port " + port + " time out.");
                return false;
            }
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                // ignore
            }
            tryCount ++;
        }
    }

    public static void startDerby(int port) throws IOException {
        Runtime runtime = Runtime.getRuntime();
        runtime.addShutdownHook(new Thread() {
            public void run() {
                if (derbyProcess != null)
                    derbyProcess.destroy();
            }
        });
        System.out.print("Start embedded database...");
        String cmd = String.format(
            "java -Dderby.drda.portNumber=%d -jar out/bin/derbyrun.jar server start -noSecurityManager",
            port
        );
        derbyProcess = runtime.exec(cmd);
        if (!CommonTools.waitForService(port)) {
            System.out.println("FAILED");
            System.exit(1);
        }
        System.out.println("OK");
    }

    public static String escapeString(String s, char enclosed) {
        StringBuilder builder = new StringBuilder();
        builder.append(enclosed);
        for (char c : s.toCharArray()) {
            if (c == enclosed)
                builder.append("\\").append(enclosed);
            else
                builder.append(c);
        }
        builder.append(enclosed);
        return builder.toString();
    }


    public static String unescapeString(String s, char enclosed) {
        // return unchanged when the string is not enclosed.
        if (s.charAt(0) != enclosed)
            return s;

        String reg = "\\\\" + enclosed;
        String replaced = s.replaceAll(reg, Character.toString(enclosed));
        return replaced.substring(1, replaced.length() - 1);
    }
}
