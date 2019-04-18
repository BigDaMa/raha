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

package qa.qcri.nadeef.console;

import com.google.common.base.Stopwatch;
import com.google.common.base.Strings;
import com.google.common.collect.Lists;
import jline.console.ConsoleReader;
import jline.console.completer.*;
import qa.qcri.nadeef.core.datamodel.CleanPlan;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.datamodel.Rule;
import qa.qcri.nadeef.core.pipeline.CleanExecutor;
import qa.qcri.nadeef.core.pipeline.UpdateExecutor;
import qa.qcri.nadeef.core.utils.Bootstrap;
import qa.qcri.nadeef.core.utils.CSVTools;
import qa.qcri.nadeef.core.utils.sql.DBInstaller;
import qa.qcri.nadeef.core.utils.sql.SQLDialectBase;
import qa.qcri.nadeef.core.utils.sql.SQLDialectFactory;
import qa.qcri.nadeef.tools.CommonTools;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.PerfReport;
import qa.qcri.nadeef.tools.Logger;
import sun.rmi.runtime.Log;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * User interactive console.
 *
 */
public class Console {

    //<editor-fold desc="Private fields">
    private static final String logo =
        "   _  __        __        _____\n" +
        "  / |/ /__ ____/ /__ ___ /   _/\n" +
        " /    / _ `/ _  / -_) -_)   _/\n" +
        "/_/|_/\\_,_/\\_,_/\\__/\\__/ __/\n" +
        "Data Cleaning solution (Build " + System.getenv("BuildVersion")  +
        ", using Java " + System.getProperty("java.version") + ").\n" +
        "Copyright (C) Qatar Computing Research Institute, 2013 - Present (http://da.qcri.org).";

    private static final String helpInfo = "Type 'help' to see what commands we have.";

    private static final String prompt = ":> ";

    private static final String[] commands =
        { "load", "run", "repair", "detect", "help", "set", "exit", "append" };
    private static ConsoleReader console;
    private static List<CleanPlan> cleanPlans;
    private static List<CleanExecutor> executors = Lists.newArrayList();
    private static Logger tracer = Logger.getLogger(Console.class);
    private static int lastExecutorIndex = -1;

    //</editor-fold>

    //<editor-fold desc="Detect Thread class">
    /**
     * Detect thread.
     */
    private static class DetectRunnable implements Runnable {
        private CleanExecutor executor;
        public DetectRunnable(CleanExecutor executor) {
            this.executor = executor;
        }

        @Override
        public void run() {
            executor.detect();
        }
    }
    //</editor-fold>

    //<editor-fold desc="Repair Thread class">

    /**
     * Repair thread class.
     */
    private static class RepairRunnable implements Runnable {
        private CleanExecutor executor;
        public RepairRunnable(CleanExecutor cleanExecutor) {
            this.executor = cleanExecutor;
        }

        @Override
        public void run() {
            executor.repair();
        }
    }

    //</editor-fold>

    //<editor-fold desc="Clean Thread class">

    /**
     * Repair thread class.
     */
    private static class CleanRunnable implements Runnable {
        private CleanExecutor executor;
        public CleanRunnable(CleanExecutor cleanExecutor) {
            this.executor = cleanExecutor;
        }

        @Override
        public void run() {
            executor.detect();
            executor.repair();
        }
    }

    //</editor-fold>
    /**
     * Start of Console.
     * @param args user input.
     */
    public static void main(String[] args) {
        try {
            // bootstrap Nadeef.
            Stopwatch stopwatch = Stopwatch.createStarted();
            Bootstrap.start();

            console = new ConsoleReader();
            List<Completer> loadCompleter =
                Arrays.asList(
                    new StringsCompleter(commands),
                    new FileNameCompleter(),
                    new NullCompleter()
                );
            console.addCompleter(new ArgumentCompleter(loadCompleter));

            console.clearScreen();
            console.println(logo);
            console.println();
            console.println(helpInfo);
            console.println();
            console.drawLine();
            console.setPrompt(prompt);
            console.println(
                "Your NADEEF started in " + stopwatch.elapsed(TimeUnit.MILLISECONDS) + " ms."
            );

            String line;
            while ((line = console.readLine()) != null) {
                line = line.trim();
                String[] tokens = line.split(" ");

                if (tokens.length == 0) {
                    continue;
                }

                // clear the statistics for every run.
                PerfReport.clear();
                try {
                    if (tokens[0].equalsIgnoreCase("exit")) {
                        break;
                    } else if (tokens[0].equalsIgnoreCase("load")) {
                        load(line);
                    } else if (tokens[0].equalsIgnoreCase("list")) {
                        list();
                    } else if (tokens[0].equalsIgnoreCase("help")) {
                        printHelp();
                    } else if (tokens[0].equalsIgnoreCase("detect")) {
                        detect(line);
                    } else if (tokens[0].equalsIgnoreCase("repair")) {
                        repair(line);
                    } else if (tokens[0].equalsIgnoreCase("run")) {
                        run(line);
                    } else if (tokens[0].equalsIgnoreCase("append")) {
                        append(line);
                    } else if (tokens[0].equalsIgnoreCase("set")) {
                        set(line);
                    } else if (!Strings.isNullOrEmpty(tokens[0])) {
                        console.println("I don't know this command.");
                    }
                } catch (Exception ex) {
                    console.println(
                        "Oops, something is wrong. Please check the log in the output dir."
                    );

                    tracer.error("", ex);
                }
            }
        } catch (Exception ex) {
            try {
                tracer.error("Bootstrap failed", ex);
            } catch (Exception ignore) {}
        } finally {
            Bootstrap.shutdown();
        }

        System.exit(0);
    }

    private static void load(String cmdLine) throws IOException {
        Stopwatch stopwatch = Stopwatch.createStarted();
        String[] splits = cmdLine.split("\\s");
        if (splits.length != 2) {
            console.println("Invalid load command. Run load <Nadeef config file>.");
            return;
        }
        String fileName = splits[1];
        File file = CommonTools.getFile(fileName);

        // shutdown existing executors
        for (CleanExecutor executor : executors) {
            executor.shutdown();
        }
        executors.clear();

        FileReader reader = null;
        try {
            reader = new FileReader(file);
            DBConfig dbConfig = NadeefConfiguration.getDbConfig();
            cleanPlans = CleanPlan.create(reader, dbConfig);
            for (CleanPlan cleanPlan : cleanPlans) {
                executors.add(new CleanExecutor(cleanPlan, dbConfig));
            }
        } catch (Exception ex) {
            tracer.error("Loading CleanPlan failed.", ex);
            return;
        } finally {
            if (reader != null)
                reader.close();
        }

        console.println(
            cleanPlans.size()
                + " rules loaded in "
                + stopwatch.elapsed(TimeUnit.MILLISECONDS) + " ms."
        );
        stopwatch.stop();
    }

    private static void append(String cmdLine) throws Exception {
        String[] splits = cmdLine.split("\\s");
        int defaultTableIndex = 0;
        if (splits.length != 2 && splits.length != 3) {
            console.println("Invalid append command. Run append <new data file> [table index].");
            return;
        }

        if (lastExecutorIndex == -1) {
            console.println("There is no detection just executed.");
            return;
        }

        if (splits.length == 3) {
            defaultTableIndex = Integer.parseInt(splits[2]);
        }

        String fileName = splits[1];
        File file = CommonTools.getFile(fileName);
        CleanPlan cleanPlan = cleanPlans.get(lastExecutorIndex);
        DBConfig dbConfig = cleanPlan.getSourceDBConfig();
        Rule rule = cleanPlan.getRule();
        String tableName = (String)rule.getTableNames().get(defaultTableIndex);
        SQLDialectBase dialectManager =
            SQLDialectFactory.getDialectManagerInstance(dbConfig.getDialect());

        HashSet<Integer> newTuples = CSVTools.append(dbConfig, dialectManager, tableName, file);
        executors.get(lastExecutorIndex).incrementalAppend(tableName, newTuples);
    }

    private static void list() throws IOException {
        if (cleanPlans == null) {
            console.println("There is no rule loaded.");
            return;
        }

        console.println("There are " + cleanPlans.size() + " rules loaded.");
        for (int i = 0; i < cleanPlans.size(); i ++) {
            console.println("\t" + i + ": " + cleanPlans.get(i).getRule().getRuleName());
        }
    }

    private static void detect(String cmd) throws IOException, InterruptedException {
        String[] tokens = cmd.split("\\s");
        if (tokens.length > 2) {
            console.println("Wrong detect command. Run detect [id number] instead.");
            return;
        }

        if (executors == null || executors.size() == 0) {
            console.println("There is no rule loaded.");
            return;
        }

        int index = -1;
        lastExecutorIndex = -1;
        if (tokens.length == 2) {
            index = Integer.valueOf(tokens[1]);
            if (index < 0 || index >= cleanPlans.size()) {
                console.println("Out of index.");
                return;
            }
            lastExecutorIndex = index;
        }

        if (executors.size() == 1) {
            lastExecutorIndex = 0;
        }

        for (int i = 0; i < executors.size(); i ++) {
            if (index != -1 && i != index) {
                continue;
            }

            CleanExecutor executor = executors.get(i);
            Thread thread = new Thread(new DetectRunnable(executor));
            thread.start();

            do {
                Thread.sleep(1000);
                double percentage = executor.getDetectProgress();
                printProgress(percentage, "DETECT");
            } while (thread.isAlive());

            // print out the final result.
            String ruleName = executor.getCleanPlan().getRule().getRuleName();
            double percentage = executor.getDetectProgress();
            printProgress(percentage, "DETECT");
            console.println();
            console.flush();
            tracer.info(PerfReport.generateDetectSummary(ruleName));
        }
    }

    private static void repair(String cmd) throws IOException, InterruptedException {
        String[] tokens = cmd.split("\\s");
        if (tokens.length > 2) {
            console.println("Wrong repair command. Run repair [id number] instead.");
        }

        if (executors == null || executors.size() == 0) {
            console.println("There is no rule loaded.");
            return;
        }

        int index = -1;
        if (tokens.length == 2) {
            index = Integer.valueOf(tokens[1]);
            if (index < 0 && index >= cleanPlans.size()) {
                console.println("Out of index.");
                return;
            }
        }

        for (int i = 0; i < executors.size(); i ++) {
            if (index != -1 && index != i) {
                continue;
            }

            CleanExecutor executor = executors.get(i);
            Thread thread = new Thread(new RepairRunnable(executor));
            thread.start();

            do {
                Thread.sleep(1000);
                double percentage = executor.getRepairProgress();
                printProgress(percentage, "REPAIR");
            } while (thread.isAlive());

            // print out the final result.
            String ruleName = executor.getCleanPlan().getRule().getRuleName();
            double percentage = executor.getRepairProgress();
            printProgress(percentage, "REPAIR");
            console.println();
            console.flush();
            tracer.info(PerfReport.generateRepairSummary(ruleName));
        }
    }

    private static void run(String cmd)
        throws IOException, InterruptedException {
        String[] tokens = cmd.split("\\s");
        if (tokens.length > 2) {
            console.println("Wrong repair command. Run repair [id number] instead.");
        }

        if (executors == null || executors.size() == 0) {
            console.println("There is no rule loaded.");
            return;
        }

        int index = -1;
        if (tokens.length == 2) {
            index = Integer.valueOf(tokens[1]);
            if (index < 0 && index >= cleanPlans.size()) {
                console.println("Out of index.");
                return;
            }
        }

        // TODO: Here the updater only has one source connection, it is wrong since
        // a update can be in multiple sources from different DB. Think about a pattern
        // to fix it.
        UpdateExecutor updateExecutor =
            new UpdateExecutor(
                cleanPlans.get(0),
                NadeefConfiguration.getDbConfig()
            );
        int updatedCell = 0;
        int maxIterationNumber = 0;

        do {
            try {
                DBInstaller.cleanExecutionDB();
            } catch (Exception ex) {
                tracer.error("Cleaning database failed.", ex);
            }
            for (int i = 0; i < executors.size(); i ++) {
                if (index != -1 && index != i) {
                    continue;
                }

                CleanExecutor executor = executors.get(i);
                Thread thread = new Thread(new CleanRunnable(executor));
                thread.start();

                do {
                    Thread.sleep(1000);
                    double percentage = executor.getRepairProgress();
                    printProgress(percentage, "CLEAN");
                } while (thread.isAlive());

                // print out the final result.
                double percentage = executor.getRepairProgress();
                printProgress(percentage, "CLEAN");
                console.println();
                console.flush();
            }

            // do the final holistic update
            updateExecutor.run();
            updatedCell = updateExecutor.getUpdateCellCount();
            maxIterationNumber ++;
        } while (
            updatedCell != 0 &&
            maxIterationNumber <= NadeefConfiguration.getMaxIterationNumber()
        );

        // Print overall statistics
        for (int i = 0; i < executors.size(); i ++) {
            if (index != -1 && index != i) {
                continue;
            }
            CleanExecutor executor = executors.get(i);
            String ruleName = executor.getCleanPlan().getRule().getRuleName();
            tracer.info(PerfReport.generateDetectSummary(ruleName));
            tracer.info(PerfReport.generateRepairSummary(ruleName));
        }
        console.println();
        tracer.info(PerfReport.generateUpdateSummary());
    }

    private static void set(String cmd) throws IOException {
        String[] tokens = cmd.split("\\s");
        if (tokens[1].equalsIgnoreCase("alwaysCompile")) {
            boolean mode = !NadeefConfiguration.getAlwaysCompile();
            console.println("set alwaysCompile " + (mode ? "on" : "off"));
            NadeefConfiguration.setAlwaysCompile(mode);
        }
    }

    private static void printHelp() throws IOException {
        final String help =
                " |NADEEF console usage:\n" +
                " |----------------------------------\n" +
                " |help : Print out this help information.\n" +
                " |\n" +
                " |load <input CleanPlan file> :\n" +
                " |    load a NADEEF clean plan.\n" +
                " |\n" +
                " |detect [rule id] :\n" +
                " |    start the violation detection with a given rule id number.\n" +
                " |\n" +
                " |list : \n" +
                " |    list available rules.\n" +
                " |\n" +
                " |repair [rule id] :\n" +
                " |    repair the data source with a given rule id number.\n" +
                " |\n" +
                " |run [rule id]:\n" +
                " |    run both detect and repair with a given rule id number. \n" +
                " |\n" +
                " |append <new data file> [table index]:\n" +
                " |    appending new data into the source from the last detection. \n" +
                " |\n" +
                " |exit :\n" +
                " |    exit the console.\n";
        console.println(help);
    }

    //<editor-fold desc="Private helpers">
    private static void printProgress(double percentage, String title) throws IOException {
        console.redrawLine();
        int ne = (int)Math.round(percentage * 50.f);
        StringBuilder stringBuilder = new StringBuilder(512);
        stringBuilder.append('[').append(title).append("][");
        for (int i = 0; i < ne; i ++) {
            stringBuilder.append("=");
        }

        if (ne < 50) {
            stringBuilder.append(">");
            for (int i = 0; i < 50 - ne; i ++) {
                stringBuilder.append(" ");
            }
        }
        stringBuilder.append("]");
        stringBuilder.append(Math.round(percentage * 100));
        stringBuilder.append(" %");
        console.print(stringBuilder.toString());
        console.flush();
    }
    //</editor-fold>
}
