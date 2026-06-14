import {
  BadRequestException,
  Controller,
  Delete,
  Get,
  Param,
  Post,
  UploadedFiles,
  UseGuards,
  UseInterceptors,
} from '@nestjs/common';
import { FilesInterceptor } from '@nestjs/platform-express';
import {
  ApiBasicAuth,
  ApiBody,
  ApiConsumes,
  ApiOperation,
  ApiParam,
  ApiTags,
} from '@nestjs/swagger';
import { BasicAuthGuard } from '../auth/basic-auth.guard';
import { BundlesService } from './bundles.service';
import { multerOptions } from './multer.config';

// All routes here require HTTP Basic auth.
@ApiTags('bundles')
@ApiBasicAuth()
@UseGuards(BasicAuthGuard)
@Controller('api/bundles')
export class BundlesController {
  constructor(private readonly bundles: BundlesService) {}

  // Upload one or more Addressables files (bundles, catalog, hash) for a
  // given platform/profile.
  // POST /api/bundles/:platform   multipart/form-data, field name "files"
  @Post(':platform')
  @ApiOperation({
    summary: 'Upload Addressables files (bundles, catalog, hash) for a platform',
  })
  @ApiParam({
    name: 'platform',
    description: 'Platform / Addressables profile, e.g. Android, iOS, StandaloneWindows64',
    example: 'Android',
  })
  @ApiConsumes('multipart/form-data')
  @ApiBody({
    schema: {
      type: 'object',
      properties: {
        files: {
          type: 'array',
          items: { type: 'string', format: 'binary' },
        },
      },
    },
  })
  @UseInterceptors(FilesInterceptor('files', 100, multerOptions))
  upload(
    @Param('platform') platform: string,
    @UploadedFiles() files: Express.Multer.File[],
  ) {
    if (!files || files.length === 0) {
      throw new BadRequestException('No files uploaded (use field name "files")');
    }
    return {
      platform: this.bundles.platformDir(platform).safe,
      uploaded: this.bundles.describeUploaded(platform, files),
    };
  }

  // List all platforms/profiles.
  // GET /api/bundles
  @Get()
  @ApiOperation({ summary: 'List all platforms/profiles' })
  listPlatforms() {
    return this.bundles.listPlatforms();
  }

  // List stored files for a platform.
  // GET /api/bundles/:platform
  @Get(':platform')
  @ApiOperation({ summary: 'List stored files for a platform' })
  @ApiParam({ name: 'platform', example: 'Android' })
  list(@Param('platform') platform: string) {
    return this.bundles.list(platform);
  }

  // Delete a single file.
  // DELETE /api/bundles/:platform/:filename
  @Delete(':platform/:filename')
  @ApiOperation({ summary: 'Delete a single file' })
  @ApiParam({ name: 'platform', example: 'Android' })
  @ApiParam({ name: 'filename', example: 'catalog.json' })
  remove(
    @Param('platform') platform: string,
    @Param('filename') filename: string,
  ) {
    return { deleted: this.bundles.remove(platform, filename) };
  }
}
