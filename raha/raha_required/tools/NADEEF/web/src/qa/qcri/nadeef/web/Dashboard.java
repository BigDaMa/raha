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

import com.google.common.base.Strings;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;

import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;

import qa.qcri.nadeef.tools.sql.SQLDialect;
import qa.qcri.nadeef.web.rest.WidgetAction;
import qa.qcri.nadeef.web.rest.TableAction;
import qa.qcri.nadeef.web.rest.ProjectAction;
import qa.qcri.nadeef.web.rest.RuleAction;
import qa.qcri.nadeef.web.rest.SourceAction;
import qa.qcri.nadeef.web.rest.RemoteAction;
import qa.qcri.nadeef.web.rest.AnalyticAction;

import java.util.logging.Level;
import java.util.logging.Logger;

import static spark.Spark.*;

/**
 * Start class for launching dashboard.
 */
public final class Dashboard {
    private static final String defaultRoot = "qa/qcri/nadeef/web/public";
    private static final Logger logger = Logger.getLogger(Dashboard.class.getName());

    //<editor-fold desc="Where everything begins">
    public static void main(String[] args) {
        Bootstrap.start();

        String rootDir = System.getProperty("rootDir");
        if (Strings.isNullOrEmpty(rootDir)) {
            logger.info("Set rootDir " + defaultRoot);
            staticFileLocation(defaultRoot);
        } else {
            logger.info("Set rootDir " + rootDir);
            externalStaticFileLocation(rootDir);
        }

        get("/", (request, response) -> {
            response.redirect("/index.html");
            return 0;
        });

        exception(Exception.class, (ex, request, response) -> {
            response.status(400);
            logger.log(Level.SEVERE, "Exception happens ", ex);
            JsonObject json = new JsonObject();
            json.add("error", new JsonPrimitive(ex.getMessage()));
            response.body(json.toString());
        });

        SQLDialect dialect = NadeefConfiguration.getDbConfig().getDialect();
        WidgetAction.setup(dialect);
        TableAction.setup(dialect);
        RuleAction.setup(dialect);
        SourceAction.setup(dialect);
        RemoteAction.setup(dialect);
        ProjectAction.setup(dialect);
        AnalyticAction.setup(dialect);
    }
}