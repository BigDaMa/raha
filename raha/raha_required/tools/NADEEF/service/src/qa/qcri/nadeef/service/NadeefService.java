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

package qa.qcri.nadeef.service;

import com.google.common.util.concurrent.AbstractIdleService;
import org.apache.thrift.server.TServer;
import org.apache.thrift.server.TThreadPoolServer;
import org.apache.thrift.transport.TServerSocket;
import org.apache.thrift.transport.TServerTransport;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.utils.Bootstrap;
import qa.qcri.nadeef.service.thrift.TNadeefService;
import qa.qcri.nadeef.tools.Logger;

/**
 * Service container which starts / stops the NADEEF thrift service.
 */
public class NadeefService extends AbstractIdleService {
    private TServer server;
    private static Logger logger = Logger.getLogger(NadeefService.class);

    @Override
    protected void startUp() throws Exception {
        Bootstrap.start();
        int port = NadeefConfiguration.getServerPort();

        NadeefServiceHandler handler = new NadeefServiceHandler();
        TNadeefService.Processor processor =
            new TNadeefService.Processor(handler);
        TServerTransport serverTransport = new TServerSocket(port);
        server =
            new TThreadPoolServer(
                new TThreadPoolServer
                    .Args(serverTransport)
                    .processor(processor)
            );
        logger.info("Starting NADEEF server @ " + port);
        server.serve();
    }

    @Override
    protected void shutDown() throws Exception {
        if (server != null && server.isServing()) {
            server.stop();
        }
        Bootstrap.shutdown();
    }

    /**
     * Starts the NADEEF server.
     * @param args command line args.
     */
    public static void main(String[] args) {
        NadeefService service = null;
        try {
            service = new NadeefService();
            service.startUp();
            Thread.sleep(100);
        } catch (Exception ex) {
            logger.error("Nadeef service has exception underneath.", ex);
        } finally {
            if (service != null) {
                try {
                    service.shutDown();
                } catch (Exception ex) {
                    // ignore
                }
            }
            Bootstrap.shutdown();
        }
    }
}
