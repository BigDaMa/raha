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

package qa.qcri.nadeef.web;

import qa.qcri.nadeef.tools.CommonTools;
import qa.qcri.nadeef.tools.Logger;

import java.io.FileReader;

/**
 * Nadeef Dashboard launcher.
 */
public final class NadeefStart {
    private static Process thriftProcess;
    private static final int WEB_PORT = 4567;

    public static void main(String[] args) {
        Logger logger = Logger.getLogger(NadeefStart.class);

        try {
            System.getProperties().load(new FileReader("nadeef.conf"));
            Runtime runtime = Runtime.getRuntime();
            runtime.addShutdownHook(new Thread() {
                public void run() {
                    if (thriftProcess != null)
                        thriftProcess.destroy();
                }
            });

            if (CommonTools.isPortOccupied(WEB_PORT)) {
                logger.error("Web port " + WEB_PORT + " is occupied, please clear the port first.");
                System.exit(1);
            }

            System.out.print("Start thrift server...");
            if (CommonTools.isLinux() || CommonTools.isMac()) {
                thriftProcess =
                    Runtime.getRuntime().exec(
                        "java -cp out/bin/*:.:out/ qa.qcri.nadeef.service.NadeefService");
            } else {
                thriftProcess =
                    Runtime.getRuntime().exec(
                        "java -cp out/bin/*;.;out/ qa.qcri.nadeef.service.NadeefService");
            }

            int thriftPort = Integer.parseInt(System.getProperty("thrift.port", "9091"));
            if (!CommonTools.waitForService(thriftPort)) {
                System.out.println("FAILED");
                System.exit(1);
            }

            logger.info("OK");

            logger.info("NADEEF Dashboard is live at http://localhost:4567/index.html");
            Dashboard.main(args);
        } catch (Exception ex) {
            logger.error("Launching dashboard failed, shutdown.", ex);
        }
    }
}
