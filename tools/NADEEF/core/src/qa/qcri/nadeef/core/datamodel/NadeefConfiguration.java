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

package qa.qcri.nadeef.core.datamodel;

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import qa.qcri.nadeef.core.utils.RuleBuilder;
import qa.qcri.nadeef.tools.CommonTools;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.Logger;
import qa.qcri.nadeef.tools.sql.SQLDialectTools;

import java.io.File;
import java.io.Reader;
import java.nio.file.Path;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Properties;

/**
 * NADEEF configuration class.
 * TODO: make it auto-generated from property file.
 */
public final class NadeefConfiguration {
    private static Logger logger = Logger.getLogger(NadeefConfiguration.class);

    private static DBConfig dbConfig;
    private static HashMap<String, RuleBuilder> ruleExtension = Maps.newHashMap();
    private static Optional<Class> decisionMakerClass;
    private static Path outputPath;
    private static Properties properties;

    private static void initialize() throws Exception {
        dbConfig =
            new DBConfig.Builder()
                .url(properties.getProperty("database.url"))
                .username(properties.getProperty("database.username"))
                .password(properties.getProperty("database.password"))
                .dialect(
                    SQLDialectTools.getSQLDialect(
                        properties.getProperty("database.type", "database.derby")))
                .build();

        if (properties.containsKey("general.fixdecisionmaker")) {
            String className = properties.getProperty("general.fixdecisionmaker");
            Class customizedClass = CommonTools.loadClass(className);
            decisionMakerClass = Optional.of(customizedClass);
        } else
            decisionMakerClass = Optional.absent();

        if (properties.containsKey("general.outputPath")) {
            String outputPathString = properties.getProperty("general.outputPath");
            File tmpPath = new File(outputPathString);
            if (tmpPath.exists() && tmpPath.isDirectory())
                outputPath = tmpPath.toPath();
            else {
                String userDir = System.getProperty("user.dir");
                logger.info(String.format(
                    "Cannot find directory %s, we use %s as default.",
                    outputPathString,
                    userDir
                ));
                outputPath = new File(outputPathString).toPath();
            }
        }

        Enumeration<?> enumeration = properties.propertyNames();
        while (enumeration.hasMoreElements()) {
            String property = (String)enumeration.nextElement();
            if (property.startsWith("ruleext.")) {
                String key = property.substring("ruleext.".length());
                String builderClassName = properties.getProperty(property);
                Class builderClass = CommonTools.loadClass(builderClassName);
                @SuppressWarnings("unchecked")
                RuleBuilder writer = (RuleBuilder)(builderClass.getConstructor().newInstance());
                ruleExtension.put(key, writer);
            }
        }
    }

    /**
     * Initialize configuration from string.
     * @param reader configuration string.
     */
    public static void initialize(Reader reader) throws Exception {
        Preconditions.checkNotNull(reader);
        properties = System.getProperties();
        properties.load(reader);
        initialize();
    }

    /**
     * Gets the server url.
     * @return server url.
     */
    public static String getServerUrl() {
        return properties.getProperty("thrift.url", "localhost");
    }

    /**
     * Gets server port.
     * @return port number
     */
    public static int getServerPort() {
        return Integer.parseInt(properties.getProperty("thrift.port", "9091"));
    }

    public static void setAlwaysOverride(boolean isAlwaysOverride) {
        properties.setProperty("general.alwaysOverride", Boolean.toString(isAlwaysOverride));
    }

    public static void setMaxIterationNumber(int maxIterationNumber) {
        properties.setProperty("general.maxIterationNumber", Integer.toString(maxIterationNumber));
    }

    public static void setAlwaysCompile(boolean isAlwaysCompile) {
        properties.setProperty("general.alwaysCompile", Boolean.toString(isAlwaysCompile));
    }

    public static void setDecisionMakerClass(Class decisionMaker) {
        decisionMakerClass = Optional.of(decisionMaker);
    }

    /**
     * Gets the NADEEF output path. Output path is used for writing logs,
     * temporary class files, etc.
     * @return the NADEEF output path.
     */
    public static Path getOutputPath() {
        return outputPath;
    }

    public static int getDerbyPort() {
        return Integer.parseInt(properties.getProperty("general.derby.port", "45000"));
    }

    /**
     * Gets the {@link qa.qcri.nadeef.tools.DBConfig} of Nadeef metadata database.
     * @return meta data {@link qa.qcri.nadeef.tools.DBConfig}.
     */
    public static DBConfig getDbConfig() {
        return dbConfig;
    }

    /**
     * Try gets the rule builder from the extensions.
     * @param typeName type name.
     * @return RuleBuilder instance.
     */
    public static RuleBuilder tryGetRuleBuilder(String typeName) {
        String typeName_ = typeName.toLowerCase();
        if (ruleExtension.containsKey(typeName_)) {
            return ruleExtension.get(typeName_);
        }
        return null;
    }

    /**
     * Gets the Nadeef installed schema name.
     * @return Nadeef DB schema name.
     */
    public static int getMaxIterationNumber() {
        return Integer.parseInt(
            properties.getProperty("general.maxIterationNumber", "10")
        );
    }

    /**
     * Gets notebook URL.
     * @return Notebook URL.
     */
    public static String getNotebookUrl() {
        return properties.getProperty("notebook.url", "localhost:8888");
    }

    /**
     * Gets Nadeef violation table name.
     * @return violation table name.
     */
    public static String getViolationTableName() {
        return "VIOLATION";
    }

    /**
     * Gets Nadeef violation table name.
     * @return violation table name.
     */
    public static String getRepairTableName() {
        return "REPAIR";
    }

    /**
     * Gets AlwaysCompile option.
     * @return alwaysCompile value.
     */
    public static boolean getAlwaysCompile() {
        return Boolean.parseBoolean(
            properties.getProperty("general.alwaysCompile", "false"));
    }

    /**
     * Gets OverwriteTable option.
     * @return OverwriteTable value.
     */
    public static boolean getAlwaysOverrideTable() {
        return Boolean.parseBoolean(
            properties.getProperty("general.alwaysOverwriteTable", "false"));
    }

    /**
     * Gets the Audit table name.
     * @return audit table name.
     */
    public static String getAuditTableName() {
        return "AUDIT";
    }

    /**
     * Gets the decision maker class.
     * @return decision maker class. It is absent when user is not providing a customized
     */
    public static Optional<Class> getDecisionMakerClass() {
        return decisionMakerClass;
    }
}
