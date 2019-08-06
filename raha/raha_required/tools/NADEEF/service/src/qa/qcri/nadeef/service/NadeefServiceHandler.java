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

import com.google.common.base.Strings;
import com.google.common.collect.Lists;
import com.google.common.io.Files;
import org.apache.thrift.TException;
import qa.qcri.nadeef.core.datamodel.CleanPlan;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.datamodel.Rule;
import qa.qcri.nadeef.core.datamodel.Schema;
import qa.qcri.nadeef.core.utils.RuleBuilder;
import qa.qcri.nadeef.core.utils.sql.DBMetaDataTool;
import qa.qcri.nadeef.service.thrift.*;
import qa.qcri.nadeef.tools.CommonTools;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.Logger;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.Collection;
import java.util.List;

/**
 * NadeefServiceHandler handles request for NADEEF service.
 */
// TODO: speedup the compiling stage by using object caching.
public class NadeefServiceHandler implements TNadeefService.Iface {
    private static Logger tracer = Logger.getLogger(NadeefServiceHandler.class);

    /**
     * {@inheritDoc}
     */
    @Override
    public String generate(TRule tRule, String tableName, String dbname)
        throws TNadeefRemoteException {
        String result = "";
        String type = tRule.getType();
        String code = tRule.getCode();
        String name = tRule.getName();

        if (type.equalsIgnoreCase("udf")) {
            result = code;
        } else {
            String[] codeLines = code.split("\n");
            List<String> codes = Lists.newArrayList(codeLines);
            DBConfig dbConfig = new DBConfig(NadeefConfiguration.getDbConfig());
            dbConfig.switchDatabase(dbname);

            try {
                Schema schema =
                    DBMetaDataTool.getSchema(dbConfig, tableName);
                RuleBuilder ruleBuilder =
                    NadeefConfiguration.tryGetRuleBuilder(type);
                if (ruleBuilder == null)
                    throw new IllegalArgumentException("Type " + type + " is not supported.");

                Collection<File> javaFiles =
                    ruleBuilder
                        .name(name)
                        .schema(schema)
                        .table(tableName)
                        .value(codes)
                        .generate();
                // TODO: currently only picks the first generated file
                File codeFile = javaFiles.iterator().next();
                result = Files.toString(codeFile, Charset.defaultCharset());
            } catch (Exception ex) {
                tracer.error("Code generation failed.", ex);
                TNadeefRemoteException re = new TNadeefRemoteException();
                re.setType(TNadeefExceptionType.UNKNOWN);
                re.setMessage(ex.getMessage());
                throw re;
            }
        }
        return result;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean verify(TRule rule) throws TNadeefRemoteException {
        String type = rule.getType();
        String code = rule.getCode();
        String name = rule.getName();
        try {
            if (type.equalsIgnoreCase("udf")) {
                Path outputPath =
                    FileSystems.getDefault().getPath(
                        NadeefConfiguration.getOutputPath().toString(),
                        name + ".java"
                    );

                Files.write(
                    code.getBytes(StandardCharsets.UTF_8),
                    outputPath.toFile()
                );

                String msg = CommonTools.compileFile(outputPath.toFile());
                if (msg != null) {
                    TNadeefRemoteException ex = new TNadeefRemoteException();
                    ex.setType(TNadeefExceptionType.COMPILE_ERROR);
                    ex.setMessage(msg);
                    throw ex;
                }
            } else {
                TNadeefRemoteException ex = new TNadeefRemoteException();
                ex.setType(TNadeefExceptionType.COMPILE_ERROR);
                ex.setMessage("Rule type " + type + " is not verifiable.");
                throw ex;
            }
        } catch (Exception ex) {
            tracer.error("Exception happens in verify.", ex);
            if (ex instanceof TNadeefRemoteException)
                throw (TNadeefRemoteException)ex;
        }
        return true;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String detect(
        TRule rule,
        String table1,
        String table2,
        String outputdb
    ) throws TNadeefRemoteException {
        tracer.info("Detect rule " + rule.getName() + "[" + rule.getType() + "]");
        if (rule.getType().equals("udf") && !verify(rule)) {
            TNadeefRemoteException ex = new TNadeefRemoteException();
            ex.setType(TNadeefExceptionType.COMPILE_ERROR);
            throw ex;
        }

        List<String> tables = Lists.newArrayList();
        tables.add(table1);
        if (table2 != null && !table2.isEmpty()) {
            tables.add(table2);
        }

        try {
            NadeefJobScheduler scheduler = NadeefJobScheduler.getInstance();

            DBConfig dbConfig = new DBConfig(NadeefConfiguration.getDbConfig());
            dbConfig.switchDatabase(outputdb);

            String type = rule.getType();
            String name = rule.getName();
            String key = null;
            Rule ruleInstance;
            CleanPlan cleanPlan;
            if (type.equalsIgnoreCase("udf")) {
                String currentPath = NadeefConfiguration.getOutputPath().toString();
                String fileName = currentPath + File.separator + name + ".java";
                File outputFile = new File(fileName);

                try (FileOutputStream os = new FileOutputStream(outputFile)) {
                    String code = rule.getCode();
                    os.write(code.getBytes());
                    os.flush();
                }

                tracer.info("Loading " + fileName);
                String message = CommonTools.compileFile(outputFile);
                if (!Strings.isNullOrEmpty(message))
                    throw new Exception(message);

                Class udfClass = CommonTools.loadClass(name);
                if (!Rule.class.isAssignableFrom(udfClass)) {
                    throw new IllegalArgumentException("The specified class is not a Rule class.");
                }

                ruleInstance = (Rule) udfClass.newInstance();
                ruleInstance.initialize(rule.getName(), tables);
                cleanPlan = new CleanPlan(dbConfig, ruleInstance);
                key = scheduler.submitDetectJob(cleanPlan);
            } else {
                // TODO: declarative rule only supports 1 table
                Collection<Rule> rules =
                    buildAbstractRule(dbConfig,  rule, table1);
                for (Rule rule_ : rules) {
                    rule_.initialize(rule.getName(), tables);
                    cleanPlan = new CleanPlan(dbConfig, rule_);
                    key = scheduler.submitDetectJob(cleanPlan);
                }
            }

            return key;

        } catch (Exception ex) {
            tracer.error("Exception in detect", ex);
            TNadeefRemoteException re = new TNadeefRemoteException();
            re.setType(TNadeefExceptionType.COMPILE_ERROR);
            re.setMessage(ex.getMessage());
            throw re;
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String repair(
        TRule rule,
        String table1,
        String table2,
        String outputdb
    ) throws TNadeefRemoteException {
        if (!verify(rule)) {
            TNadeefRemoteException ex = new TNadeefRemoteException();
            ex.setType(TNadeefExceptionType.COMPILE_ERROR);
            throw ex;
        }

        List<String> tables = Lists.newArrayList();
        tables.add(table1);
        if (table2 != null && !table2.isEmpty()) {
            tables.add(table2);
        }

        try {
            String name = rule.getName();
            Class udfClass = CommonTools.loadClass(name);
            if (!Rule.class.isAssignableFrom(udfClass)) {
                throw new IllegalArgumentException("The specified class is not a Rule class.");
            }

            Rule ruleInstance = (Rule) udfClass.newInstance();
            ruleInstance.initialize(rule.getName(), tables);
            DBConfig dbConfig = NadeefConfiguration.getDbConfig();
            DBConfig config =
                new DBConfig.Builder()
                    .dialect(dbConfig.getDialect())
                    .username(dbConfig.getUserName())
                    .password(dbConfig.getPassword())
                    .url(dbConfig.getHostName(), outputdb)
                    .build();

            NadeefJobScheduler scheduler = NadeefJobScheduler.getInstance();
            String key = scheduler.submitRepairJob(new CleanPlan(config, ruleInstance));
            return key;
        } catch (Exception ex) {
            tracer.error("Exception in detect", ex);
            TNadeefRemoteException re = new TNadeefRemoteException();
            re.setType(TNadeefExceptionType.UNKNOWN);
            re.setMessage(ex.getMessage());
            throw re;
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public TJobStatus getJobStatus(String key) throws TException {
        NadeefJobScheduler jobScheduler = NadeefJobScheduler.getInstance();
        return jobScheduler.getJobStatus(key);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<TJobStatus> getAllJobStatus() throws TException {
        NadeefJobScheduler jobScheduler = NadeefJobScheduler.getInstance();
        return jobScheduler.getJobStatus();
    }

    private Collection<Rule> buildAbstractRule(
        DBConfig dbConfig,
        TRule tRule,
        String tableName
    ) throws Exception {
        String type = tRule.getType();
        String name = tRule.getName();
        String code = tRule.getCode();

        List<String> lines = Lists.newArrayList(code.split("\n"));

        RuleBuilder ruleBuilder;
        Collection<Rule> result;
        ruleBuilder = NadeefConfiguration.tryGetRuleBuilder(type);
        Schema schema = DBMetaDataTool.getSchema(dbConfig, tableName);
        if (ruleBuilder != null) {
            result = ruleBuilder
                .name(name)
                .schema(schema)
                .table(tableName)
                .value(lines)
                .build();
        } else {
            tracer.error("Unknown Rule type: " + type, null);
            throw new IllegalArgumentException("Unknown rule type");
        }
        return result;
    }
}
