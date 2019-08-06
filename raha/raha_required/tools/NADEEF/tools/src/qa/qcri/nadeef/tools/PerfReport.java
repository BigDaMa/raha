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

package qa.qcri.nadeef.tools;

import com.google.common.base.Strings;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Performance report.
 */
public class PerfReport {
    private PerfReport() {}
    private static Map<Metric, List<Long>> metrics = Maps.newHashMap();

    public enum Metric {
        // Detect time
        DetectTime,
        // Only with Detection
        DetectTimeOnly,
        // Detect tuple count
        DetectCount,
        // Detection thread count
        DetectThreadCount,
        // Detect per call time
        DetectCallTime,

        // Horizontal Scope Time
        HScopeTime,
        // Vertical Scope Time
        VScopeTime,
        // Number of tuples after Scope
        AfterScopeTuple,
        // Number of blocks after Block
        Blocks,

        // Iteration time
        IteratorTime,
        // Tuple generator
        IterationCount,

        // DB load time
        DBLoadTime,
        // DB Connection creation
        DBConnectionCount,
        // Source DB Connection request
        SourceDBConnectionCount,
        // NADEEF DB Connection request
        NadeefDBConnectionCount,
        // Index creation count
        SourceIndexCreationCount,

        // Violation export count
        ViolationExport,
        // Violation export time
        ViolationExportTime,

        // repair time
        RepairTime,
        // Repair per call time
        RepairCallTime,
        // EQ time
        EQTime,
        // Candidate fix export count
        FixExport,
        // Candidate fix export time
        FixImport,
        // Update cell number
        UpdatedCellNumber,
    }

    public static synchronized Map<Metric, List<Long>> getMetrics() {
        return Collections.unmodifiableMap(metrics);
    }

    /**
     * Puts values to the trace statistic entry.
     * @param metric type.
     * @param value value.
     */
    public static synchronized void appendMetric(Metric metric, long value) {
        if (metrics.containsKey(metric)) {
            List<Long> values = metrics.get(metric);
            values.add(value);
        } else {
            List<Long> values = Lists.newArrayList();
            values.add(value);
            metrics.put(metric, values);
        }
    }

    /**
     * Accumulate values in the trace statistic entry.
     * @param metric type.
     * @param value value.
     */
    public static synchronized void addMetric(Metric metric, long value) {
        if (!metrics.containsKey(metric)) {
            appendMetric(metric, value);
        } else {
            List<Long> values = metrics.get(metric);
            if (values.size() > 1) {
                throw new IllegalStateException(
                    "Entry " + metric + " is found more than once in the statistic dictionary."
                );
            }
            Long newValue = values.get(0) + value;
            values.set(0, newValue);
        }
    }

    public static List<Long> get(Metric metric) {
        return metrics.get(metric);
    }

    public static void clear() {
        metrics.clear();
    }

    /**
     * Print Update summary.
     */
    public static String generateUpdateSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("Update summary:");
        sb.append("\n");
        sb.append("----------------------------------------------------------------");
        sb.append("\n");
        sb.append(formatEntry(Metric.EQTime, "EQ time", "ms"));
        sb.append("\n");
        sb.append("----------------------------------------------------------------\n");

        Collection<Long> totalChangedCells = PerfReport.get(Metric.UpdatedCellNumber);

        Long totalChangedCell = 0l;
        for (Long tmp : totalChangedCells) {
            totalChangedCell += tmp;
        }
        sb.append("Update finished with " + totalChangedCell + " cells changed.\n");
        return sb.toString();
    }

    /**
     * Print Repair summary.
     * @param ruleName rule name.
     */
    public static String generateRepairSummary(String ruleName) {
        StringBuilder sb = new StringBuilder();
        sb.append("Repair summary:")
            .append("\n")
            .append("Rule: " + ruleName)
            .append("\n")
            .append("----------------------------------------------------------------")
            .append("\n")
            .append(formatEntry(Metric.RepairCallTime, "Repair perCall time", "ms"))
            .append("\n")
            .append(formatEntry(Metric.FixExport, "New Candidate Fix", ""))
            .append("\n")
            .append(formatEntry(Metric.FixImport, "Effective Candidate Fix", ""))
            .append("\n")
            .append("----------------------------------------------------------------")
            .append("\n");

        Collection<Long> totalTimes = PerfReport.get(Metric.RepairTime);

        Long totalTime = 0l;
        for (Long tmp : totalTimes) {
            totalTime += tmp;
        }

        sb.append("Repair finished in " + totalTime + " ms.\n");
        return sb.toString();
    }

    public static String generateDetectSummary(String ruleName) {
        StringBuilder sb = new StringBuilder();

        sb.append("Detection summary:");
        sb.append("\n");
        sb.append("Rule: " + ruleName);
        sb.append("\n");
        sb.append("----------------------------------------------------------------");
        sb.append("\n");
        sb.append(formatEntry(Metric.HScopeTime, "HScope time", "ms"));
        sb.append("\n");
        sb.append(formatEntry(Metric.VScopeTime, "VScope time", "ms"));
        sb.append("\n");
        sb.append(formatEntry(Metric.Blocks, "Blocks", ""));
        sb.append("\n");
        sb.append(formatEntry(Metric.IteratorTime, "Iterator time", "ms"));
        sb.append("\n");
        sb.append(formatEntry(Metric.DBLoadTime, "DB load time", "ms"));
        sb.append("\n");
        sb.append(formatEntry(Metric.DetectTime, "Detect time", "ms"));
        sb.append("\n");
        sb.append(formatEntry(Metric.DetectThreadCount, "Detect thread count", ""));
        sb.append("\n");
        sb.append(formatEntry(Metric.DetectCount, "Detect tuple count", ""));
        sb.append("\n");
        sb.append(formatEntry(Metric.ViolationExport, "Violation", ""));
        sb.append("\n");
        sb.append(formatEntry(Metric.ViolationExportTime, "Violation export time", ""));

        long totalTime = 0l;
        long totalViolation = 0l;
        Collection<Long> totalTimes = PerfReport.get(Metric.DetectTime);
        Collection<Long> totalViolations = PerfReport.get(Metric.ViolationExport);

        if (totalTimes != null) {
            for (Long tmp : totalTimes) {
                totalTime += tmp;
            }
        }

        if (totalViolations != null) {
            for (Long tmp : totalViolations) {
                totalViolation += tmp;
            }
        }

        sb.append("\n----------------------------------------------------------------\n");
        sb.append(
            "Detection finished in " + totalTime + " ms " +
                "and found " + totalViolation + " violations.\n"
        );
        return sb.toString();
    }

    private static String formatEntry(Metric metric, String prefix, String suffix) {
        String value;
        List<Long> values = PerfReport.get(metric);
        StringBuilder outputBuilder = new StringBuilder(50);
        // TODO: a quick fix
        if (values == null)
            return "";
        for (Long tmp : values) {
            outputBuilder.append(String.format("%9d", tmp));
        }
        value = outputBuilder.toString();
        if (!Strings.isNullOrEmpty(suffix)) {
            prefix = prefix + " (" + suffix + ")";
        }
        return String.format("%-40s %s", prefix, value);
    }
}
