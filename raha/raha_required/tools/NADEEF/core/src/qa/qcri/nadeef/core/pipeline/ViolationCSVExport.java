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

import com.google.common.base.Stopwatch;
import qa.qcri.nadeef.core.utils.sql.SQLDialectBase;
import qa.qcri.nadeef.core.utils.sql.SQLDialectFactory;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.PerfReport;
import qa.qcri.nadeef.tools.Logger;

import java.io.File;
import java.util.concurrent.TimeUnit;

public class ViolationCSVExport extends Operator<File, File> {
    public ViolationCSVExport(ExecutionContext context) {
        super(context);
    }

    @Override
    protected File execute(File file) throws Exception {
        Stopwatch stopwatch = Stopwatch.createStarted();
        Logger tracer = Logger.getLogger(ViolationCSVExport.class);
        DBConfig config = getCurrentContext().getConnectionPool().getNadeefConfig();
        SQLDialectBase instance =
            SQLDialectFactory.getDialectManagerInstance(config.getDialect());
        tracer.info("Load " + file.getCanonicalPath() + " into violation table");
        if (instance.supportBulkLoad()) {
            instance.bulkLoad(config, "VIOLATION", file.toPath(), false);
        } else {
            instance.fallbackLoad(config, "VIOLATION", file, false);
        }

        PerfReport.appendMetric(
            PerfReport.Metric.ViolationExportTime,
            stopwatch.elapsed(TimeUnit.MILLISECONDS)
        );
        stopwatch.stop();
        return file;
    }
}
