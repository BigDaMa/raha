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

import com.google.common.base.Strings;
import com.google.common.io.Files;
import com.google.gson.Gson;
import com.google.gson.JsonElement;

import org.apache.thrift.TException;

import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.utils.CSVTools;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.Logger;
import qa.qcri.nadeef.tools.sql.SQLDialect;
import qa.qcri.nadeef.web.Bootstrap;
import qa.qcri.nadeef.web.NadeefClient;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.StringReader;
import java.io.BufferedWriter;

import static spark.Spark.get;
import static spark.Spark.post;

public class RemoteAction {
    private static Logger logger = Logger.getLogger(RemoteAction.class);

    //<editor-fold desc="Do actions">
    public static void setup(SQLDialect dialect) {
        NadeefClient nadeefClient = Bootstrap.getNadeefClient();
        String TABLE_PREFIX = "TB_";

        get("/progress", (request, response) -> {
            try {
                String msg = nadeefClient.getJobStatus();
                return new Gson().fromJson(msg, JsonElement.class).getAsJsonObject();
            } catch (TException ex) {
                logger.error("Thrift exception", ex);
                throw new RuntimeException(ex);
            }
        });

        post("/do/generate", (request, response) -> {
            try {
                String type = request.queryParams("type");
                String name = request.queryParams("name");
                String code = request.queryParams("code");
                String table1 = request.queryParams("table1");
                String project = request.queryParams("project");

                if (Strings.isNullOrEmpty(type) ||
                    Strings.isNullOrEmpty(name) ||
                    Strings.isNullOrEmpty(code) ||
                    Strings.isNullOrEmpty(table1) ||
                    Strings.isNullOrEmpty(project))
                    throw new IllegalArgumentException("Input is not valid.");

                String msg =
                    nadeefClient.generate(type, name, code, table1, project);
                return new Gson().fromJson(msg, JsonElement.class).getAsJsonObject();
            } catch (TException ex) {
                logger.error("generate failed", ex);
                throw new RuntimeException(ex);
            }
        });

        post("/do/verify", (request, response) -> {
            String type = request.queryParams("type");
            String name = request.queryParams("name");
            String code = request.queryParams("code");
            String table1 = request.queryParams("table1");

            if (Strings.isNullOrEmpty(type) ||
                Strings.isNullOrEmpty(name) ||
                Strings.isNullOrEmpty(code) ||
                Strings.isNullOrEmpty(table1))
                throw new IllegalArgumentException("Input is not valid.");

            try {
                String msg = nadeefClient.verify(type, name, code);
                return new Gson().fromJson(msg, JsonElement.class).getAsJsonObject();
            } catch (TException ex) {
                logger.error("verify failed", ex);
                throw new RuntimeException(ex);
            }
        });

        post("/do/detect", (request, response) -> {
            String type = request.queryParams("type");
            String name = request.queryParams("name");
            String code = request.queryParams("code");
            String table1 = request.queryParams("table1");
            String table2 = request.queryParams("table2");
            String dbname = request.queryParams("project");

            if (Strings.isNullOrEmpty(type) ||
                Strings.isNullOrEmpty(name) ||
                Strings.isNullOrEmpty(code) ||
                Strings.isNullOrEmpty(dbname) ||
                Strings.isNullOrEmpty(table1))
                throw new IllegalArgumentException("Input is not valid.");

            try {
                String msg =
                    nadeefClient.detect(type, name, code, table1, table2, dbname);
                return new Gson().fromJson(msg, JsonElement.class).getAsJsonObject();
            } catch (TException ex) {
                logger.error("detect failed", ex);
                throw new RuntimeException(ex);
            }
        });

        post("/do/upload", (request, response) -> {
            String body = request.body();
            BufferedReader reader = new BufferedReader(new StringReader(body));
            BufferedWriter writer = null;
            String line;
            try {
                // parse the project name
                String projectName = null;
                while ((line = reader.readLine()) != null) {
                    if (line.contains("project")) {
                        while ((line = reader.readLine()) != null) {
                            if (!line.isEmpty()) {
                                projectName = line.trim();
                                break;
                            }
                        }
                    }

                    if (projectName != null) {
                        break;
                    }
                }

                // parse the schema
                String schema = null;
                while ((line = reader.readLine()) != null) {
                    if (line.contains("schema")) {
                        while ((line = reader.readLine()) != null) {
                            if (!line.isEmpty()) {
                                schema = line.trim();
                                break;
                            }
                        }
                    }

                    if (schema != null) {
                        break;
                    }
                }

                // parse the file name
                int begin;
                String fileName = null;
                while ((line = reader.readLine()) != null) {
                    String fileNamePrefix = "filename=";
                    if (line.contains(fileNamePrefix)) {
                        begin = line.indexOf(fileNamePrefix) + fileNamePrefix.length() + 1;
                        for (int i = begin; i < line.length(); i ++) {
                            if (line.charAt(i) == '\"') {
                                fileName = line.substring(begin, i);
                                break;
                            }
                        }
                    }

                    if (fileName != null) {
                        break;
                    }
                }

                // parse until the beginning of the file
                while ((line = reader.readLine()) != null) {
                    if (line.isEmpty()) {
                        break;
                    }
                }

                // write to disk
                File outputFile = File.createTempFile(TABLE_PREFIX, fileName);
                writer = new BufferedWriter(new FileWriter(outputFile));
                while ((line = reader.readLine()) != null) {
                    if (line.contains("multipartformboundary") || line.isEmpty()) {
                        continue;
                    }
                    writer.write(line);
                    writer.write("\n");
                }
                writer.flush();
                logger.info("Write upload file to " + outputFile.getAbsolutePath());

                // dump into database
                DBConfig dbConfig = new DBConfig(NadeefConfiguration.getDbConfig());
                dbConfig.switchDatabase(projectName);

                // TODO: resolve this by using injection deps.
                qa.qcri.nadeef.core.utils.sql.SQLDialectBase dialectBase =
                    qa.qcri.nadeef.core.utils.sql.SQLDialectBase.createDialectBaseInstance(
                        dbConfig.getDialect()
                    );

                String tableName = Files.getNameWithoutExtension(fileName);
                CSVTools.dump(
                    dbConfig,
                    dialectBase,
                    outputFile,
                    tableName,
                    schema,
                    NadeefConfiguration.getAlwaysOverrideTable());

            } catch (Exception ex) {
                logger.error("Upload file failed.", ex);
                response.status(400);
                throw new RuntimeException(ex);
            } finally {
                try {
                    reader.close();
                    if (writer != null) {
                        writer.close();
                    }
                } catch (Exception ex) {}
            }
            return 0;
        });

        post("/do/repair", (request, response) -> {
            String type = request.queryParams("type");
            String name = request.queryParams("name");
            String code = request.queryParams("code");
            String table1 = request.queryParams("table1");
            String table2 = request.queryParams("table2");
            String dbName = request.queryParams("project");

            if (Strings.isNullOrEmpty(type) ||
                Strings.isNullOrEmpty(name) ||
                Strings.isNullOrEmpty(code) ||
                Strings.isNullOrEmpty(dbName) ||
                Strings.isNullOrEmpty(table1)) {
                throw new IllegalArgumentException("Input is not valid.");
            }

            try {
                String msg =
                    nadeefClient.repair(type, name, code, table1, table2, dbName);
                return new Gson().fromJson(msg, JsonElement.class).getAsJsonObject();
            } catch (TException ex) {
                logger.error("repair failed.", ex);
                throw new RuntimeException(ex);
            }
        });
    }
    //</editor-fold>

}
