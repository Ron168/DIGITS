# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import flask

from .forms import MultiImageDatasetForm
from .job import MultiImageDatasetJob
from digits import utils
from digits.dataset import tasks
from digits.webapp import scheduler
from digits.utils.forms import fill_form_if_cloned, save_form_to_job
from digits.utils.routing import request_wants_json

blueprint = flask.Blueprint(__name__, __name__)


@blueprint.route('/new', methods=['GET'])
@utils.auth.requires_login
def new():
    """
    Returns a form for a new MultiImageDatasetJob
    """
    form = MultiImageDatasetForm()

    # Is there a request to clone a job with ?clone=<job_id>
    fill_form_if_cloned(form)

    return flask.render_template('datasets/images/multi/new.html', form=form)


@blueprint.route('.json', methods=['POST'])
@blueprint.route('', methods=['POST'], strict_slashes=False)
@utils.auth.requires_login(redirect=False)
def create():
    """
    Creates a new MultiImageDatasetJob

    Returns JSON when requested: {job_id,name,status} or {errors:[]}
    """
    form = MultiImageDatasetForm()

    # Is there a request to clone a job with ?clone=<job_id>
    fill_form_if_cloned(form)

    if not form.validate_on_submit():
        if request_wants_json():
            return flask.jsonify({'errors': form.errors}), 400
        else:
            return flask.render_template('datasets/images/multi/new.html', form=form), 400

    job = None
    try:
        job = MultiImageDatasetJob(
            username=utils.auth.get_username(),
            name=form.dataset_name.data,
            group=form.group_name.data,
            mean_file_0=form.prebuilt_mean_file_0.data.strip(),
            mean_file_1=form.prebuilt_mean_file_1.data.strip(),
        )

        if form.method.data == 'prebuilt':
            pass
        else:
            raise ValueError('method not supported')

        force_same_shape = form.force_same_shape.data

        job.tasks.append(
            tasks.AnalyzeDbTask(
                job_dir=job.dir(),
                database=form.prebuilt_train_images_0.data,
                purpose=form.prebuilt_train_images_0.label.text,
                force_same_shape=force_same_shape,
            )
        )

        job.tasks.append(
            tasks.AnalyzeDbTask(
                job_dir=job.dir(),
                database=form.prebuilt_train_images_1.data,
                purpose=form.prebuilt_train_images_1.label.text,
                force_same_shape=force_same_shape,
            )
        )

        if form.prebuilt_train_labels.data:
            job.tasks.append(
                tasks.AnalyzeDbTask(
                    job_dir=job.dir(),
                    database=form.prebuilt_train_labels.data,
                    purpose=form.prebuilt_train_labels.label.text,
                    force_same_shape=force_same_shape,
                )
            )

        if form.prebuilt_val_images_0.data:
            job.tasks.append(
                tasks.AnalyzeDbTask(
                    job_dir=job.dir(),
                    database=form.prebuilt_val_images_0.data,
                    purpose=form.prebuilt_val_images_0.label.text,
                    force_same_shape=force_same_shape,
                )
            )
            job.tasks.append(
                tasks.AnalyzeDbTask(
                    job_dir=job.dir(),
                    database=form.prebuilt_val_images_1.data,
                    purpose=form.prebuilt_val_images_1.label.text,
                    force_same_shape=force_same_shape,
                )
            )
            if form.prebuilt_val_labels.data:
                job.tasks.append(
                    tasks.AnalyzeDbTask(
                        job_dir=job.dir(),
                        database=form.prebuilt_val_labels.data,
                        purpose=form.prebuilt_val_labels.label.text,
                        force_same_shape=force_same_shape,
                    )
                )

        # Save form data with the job so we can easily clone it later.
        save_form_to_job(job, form)

        scheduler.add_job(job)

        if request_wants_json():
            return flask.jsonify(job.json_dict())
        else:
            return flask.redirect(flask.url_for('digits.dataset.views.show', job_id=job.id()))

    except:
        if job:
            scheduler.delete_job(job)
        raise


def show(job, related_jobs=None):
    """
    Called from digits.dataset.views.datasets_show()
    """
    return flask.render_template('datasets/images/multi/show.html', job=job, related_jobs=related_jobs)


def summary(job):
    """
    Return a short HTML summary of a MultiImageDatasetJob
    """
    return flask.render_template('datasets/images/multi/summary.html',
                                 dataset=job)
