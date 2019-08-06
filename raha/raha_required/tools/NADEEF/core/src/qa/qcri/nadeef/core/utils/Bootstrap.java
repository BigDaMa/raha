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

package qa.qcri.nadeef.core.utils;

import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.pipeline.NodeCacheManager;
import qa.qcri.nadeef.tools.CommonTools;
import qa.qcri.nadeef.tools.Logger;
import qa.qcri.nadeef.tools.sql.SQLDialect;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.nio.file.Path;

/**
 * Bootstrapping prepares runtime environment for NADEEF. It is invoked before NADEEF starts
 * and also after NADEEF exits.
 */
public class Bootstrap {
    private static boolean isStarted;
    private static final String configurationFile = "nadeef.conf";

    private Bootstrap() {}

    /**
     * Shutdown NADEEF.
     */
    public static synchronized void shutdown() {
        if (isStarted) {
            NodeCacheManager cacheManager = NodeCacheManager.getInstance();
            cacheManager.clear();
            // try to collect the resources if possible
            System.gc();
            isStarted = false;
        }
    }

    /**
     * Bootstrap NADEEF. It tries to load the NADEEF configuration file and install
     * metadata tables in the NADEEF database.
     */
    public static synchronized boolean start() throws Exception {
        return start(configurationFile);
    }

    /**
     * Bootstrap NADEEF. It tries to load the NADEEF configuration file and install
     * metadata tables in the NADEEF database.
     * @param configFile NADEEF config file.
     */
    public static synchronized boolean start(String configFile) throws Exception {
        Preconditions.checkArgument(!Strings.isNullOrEmpty(configFile));
        if (isStarted) {
            // Here the logging is not yet started.
            System.out.println("Nadeef is already started.");
            return true;
        }

        try {
            NadeefConfiguration.initialize(new FileReader(configFile));

            // set the logging directory
            Path outputPath = NadeefConfiguration.getOutputPath();
            Logger tracer = Logger.getLogger(Bootstrap.class);
            tracer.fine("Tracer initialized at " + outputPath.toString());

            // start embedded database
            if (NadeefConfiguration.getDbConfig().getDialect() == SQLDialect.DERBY ||
                NadeefConfiguration.getDbConfig().getDialect() == SQLDialect.DERBYMEMORY)
                CommonTools.startDerby(NadeefConfiguration.getDerbyPort());
        } catch (FileNotFoundException ex) {
            System.err.println("Nadeef configuration is not found.");
            ex.printStackTrace();
        } catch (Exception ex) {
            System.err.println("NADEEF initialization failed, please check the log for detail.");
            ex.printStackTrace();
            throw ex;
        }

        isStarted = true;
        return true;
    }
}
