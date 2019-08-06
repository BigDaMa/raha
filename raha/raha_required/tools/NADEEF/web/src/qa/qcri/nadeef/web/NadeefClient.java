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

import com.google.common.base.Preconditions;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import org.apache.thrift.TException;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TProtocol;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TTransport;
import org.apache.thrift.transport.TTransportException;
import qa.qcri.nadeef.service.thrift.TJobStatus;
import qa.qcri.nadeef.service.thrift.TNadeefService;
import qa.qcri.nadeef.service.thrift.TRule;

import java.util.List;

/**
 * Nadeef Thrift client. This is the wrapper class for calling thrift methods.
 */
public final class NadeefClient {
    private static NadeefClient instance;
    private static String url;
    private static int port;

    /**
     * Initialize the Client params.
     * @param url_ remote url.
     * @param port_ port number.
     */
    public static void initialize(String url_, int port_) {
        Preconditions.checkArgument(url == null && port == 0);
        url = Preconditions.checkNotNull(url_);
        port = port_;
    }

    /**
     * Gets the instance of client.
     * @return instance.
     */
    public synchronized static NadeefClient getInstance() throws TTransportException {
        Preconditions.checkNotNull(url);
        if (instance == null) {
            instance = new NadeefClient();
        }
        return instance;
    }

    private NadeefClient() {}

    public String generate(
        String type,
        String name,
        String code,
        String table1,
        String dbname
    ) throws TException {
        TTransport transport = new TSocket(url, port);
        transport.open();
        TProtocol protocol = new TBinaryProtocol(transport);
        TNadeefService.Client client = new TNadeefService.Client(protocol);

        TRule rule = new TRule(name, type, code);
        JsonObject result = new JsonObject();
        String gen = client.generate(rule, table1, dbname);
        result.add("data", new JsonPrimitive(gen));

        transport.close();
        return result.toString();
    }

    public String verify(String type, String name, String code) throws TException {
        TTransport transport = new TSocket(url, port);
        transport.open();
        TProtocol protocol = new TBinaryProtocol(transport);
        TNadeefService.Client client = new TNadeefService.Client(protocol);
        JsonObject json = new JsonObject();
        try {
            boolean result = client.verify(new TRule(name, type, code));
            json.add("data", new JsonPrimitive(result));
        } finally {
            transport.close();
        }

        return json.toString();
    }

    public String detect(
        String type,
        String name,
        String code,
        String table1,
        String table2,
        String dbname
    ) throws TException {
        TTransport transport = new TSocket(url, port);
        transport.open();
        TProtocol protocol = new TBinaryProtocol(transport);
        TNadeefService.Client client = new TNadeefService.Client(protocol);

        TRule rule = new TRule(name, type, code);
        JsonObject json = new JsonObject();
        String result = client.detect(rule, table1, table2, dbname);
        json.add("data", new JsonPrimitive(result));

        transport.close();
        return json.toString();
    }

    public String repair(
        String type,
        String name,
        String code,
        String table1,
        String table2,
        String dbname
    ) throws TException {
        TTransport transport = new TSocket(url, port);
        transport.open();
        TProtocol protocol = new TBinaryProtocol(transport);
        TNadeefService.Client client = new TNadeefService.Client(protocol);

        TRule rule = new TRule(name, type, code);
        JsonObject result = new JsonObject();
        result.add("data", new JsonPrimitive(client.repair(rule, table1, table2, dbname)));

        transport.close();
        return result.toString();
    }

    public String getJobStatus() throws TException {
        TTransport transport = new TSocket(url, port);
        transport.open();
        TProtocol protocol = new TBinaryProtocol(transport);
        TNadeefService.Client client = new TNadeefService.Client(protocol);

        JsonObject result = new JsonObject();
        List<TJobStatus> statusList = client.getAllJobStatus();
        JsonArray jsonArray = new JsonArray();

        for (TJobStatus status : statusList) {
            JsonObject obj = new JsonObject();
            obj.add("status", new JsonPrimitive(status.getStatus().toString()));
            obj.add("overallProgress", new JsonPrimitive(status.getOverallProgress()));
            obj.add("key", new JsonPrimitive(status.getKey()));
            JsonArray array = new JsonArray();
            for (Integer progress : status.getProgress())
                array.add(new JsonPrimitive(progress));
            obj.add("progress", array);
            // Add progress stage
            // obj.add("name", status.getNames());
            jsonArray.add(obj);
        }

        result.add("data", jsonArray);
        transport.close();
        return result.toString();
    }
}
