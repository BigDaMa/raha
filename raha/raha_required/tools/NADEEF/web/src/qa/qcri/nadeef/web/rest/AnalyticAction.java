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

package qa.qcri.nadeef.web.rest;

import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.tools.Logger;
import qa.qcri.nadeef.tools.sql.SQLDialect;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

import static spark.Spark.get;
import static spark.Spark.post;

public class AnalyticAction {
    private static String getPysparkUrl() {
        String url = NadeefConfiguration.getNotebookUrl();
        return "http://" + url + "/api/notebooks";
    }
    private static Logger tracer = Logger.getLogger(AnalyticAction.class);

    public static void setup(SQLDialect dialect) {
        post("/analytic/:project", (request, response) -> {
            response.type("application/json");
            String projectName = request.params("project");
            HttpURLConnection conn = null;

            try {
                URL url = new URL(getPysparkUrl() + "/" + projectName);
                conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("PUT");
                if (conn.getResponseCode() != 201) {
                    throw new RuntimeException("Failed : HTTP error code : "
                        + conn.getResponseCode());
                }
            } catch (Exception ex) {
                tracer.error(ex.getMessage(), ex);
            } finally {
                if (conn != null)
                    conn.disconnect();
            }
            return 0;
        });

        get("/analytic/notebooks", (request, response) -> {
            response.type("application/json");
            HttpURLConnection conn = null;
            try {
                URL url = new URL(getPysparkUrl());
                conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("GET");
                if (conn.getResponseCode() != 200) {
                    throw new RuntimeException("Failed : HTTP error code : "
                        + conn.getResponseCode());
                }

                BufferedReader br = new BufferedReader(new InputStreamReader(
                    (conn.getInputStream())));
                String output = br.readLine();
                return output;
            } catch (Exception ex) {
                ex.printStackTrace();
            } finally {
                if (conn != null)
                    conn.disconnect();
            }
            return 0;
        });
    }
}
