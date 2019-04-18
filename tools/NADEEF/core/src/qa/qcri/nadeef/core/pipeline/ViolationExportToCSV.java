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

package qa.qcri.nadeef.core.pipeline;

import qa.qcri.nadeef.core.datamodel.Cell;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.datamodel.Violation;
import qa.qcri.nadeef.tools.CommonTools;
import qa.qcri.nadeef.tools.PerfReport;
import qa.qcri.nadeef.tools.Logger;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.file.Path;
import java.util.Collection;

/**
 * Exports Violation list to a CSV file
 */
public class ViolationExportToCSV extends Operator<java.util.Iterator<Violation>, File> {
    public ViolationExportToCSV(ExecutionContext context) {
        super(context);
    }

    @Override protected File execute(java.util.Iterator<Violation> violations) throws Exception {
        Logger tracer = Logger.getLogger(ViolationExportToCSV.class);
        Path outputPath = NadeefConfiguration.getOutputPath();
        int vid = 0;

        String filename =
            String.format("violation_%s_%d.csv",
                getCurrentContext().getConnectionPool().getSourceDBConfig().getDatabaseName(),
                System.currentTimeMillis()
            );

        File file = new File(outputPath.toFile(), filename);
        tracer.info("Export to " + file.getAbsolutePath());
        byte[] result = null;
        int size = 0;
        try (
            FileOutputStream fs = new FileOutputStream(file);
            BufferedOutputStream output = new BufferedOutputStream(fs);
        ) {
            while (violations.hasNext()) {
                Violation violation = violations.next();
                size ++;
                Collection<Cell> cells = violation.getCells();
                for (Cell cell : cells) {
                    StringBuffer line = new StringBuffer();
                    if (cell.hasColumnName("tid"))
                        continue;
                    String value = cell.getValue() == null ? "" : cell.getValue().toString();
                    line
                        .append(vid)
                        .append(",")
                        .append(
                            CommonTools.escapeString(
                                violation.getRuleId(),
                                CommonTools.DOUBLE_QUOTE
                            )).append(",")
                        .append(
                            CommonTools.escapeString(
                                cell.getColumn().getTableName(),
                                CommonTools.DOUBLE_QUOTE
                            )).append(",")
                        .append(cell.getTid())
                        .append(",")
                        .append(
                            CommonTools.escapeString(
                                cell.getColumn().getColumnName(),
                                CommonTools.DOUBLE_QUOTE
                            )).append(",")
                        .append(CommonTools.escapeString(value, CommonTools.DOUBLE_QUOTE))
                        .append("\n");
                    byte[] bytes = line.toString().getBytes();
                    output.write(bytes);
                }
                vid ++;
            }

            PerfReport.appendMetric(
                PerfReport.Metric.ViolationExport,
                size
            );
        }
        return file;
    }
}
